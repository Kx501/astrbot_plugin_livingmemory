# -*- coding: utf-8 -*-
"""
server.py - LivingMemory WebUI backend
Provides authentication, memory browsing, detail view and bulk deletion APIs.
Rewritten with aiohttp for better async lifecycle management.
"""

import asyncio
import json
import secrets
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, TYPE_CHECKING

from aiohttp import web
from astrbot.api import logger

from ..storage.memory_storage import MemoryStorage

if TYPE_CHECKING:
    from ..storage.faiss_manager import FaissManager
    from ..main import SessionManager


class WebUIServer:
    """
    WebUI 服务器，基于 aiohttp 实现。
    提供记忆管理的 Web 界面和 RESTful API。
    """

    def __init__(
        self,
        config: Dict[str, Any],
        faiss_manager: "FaissManager",
        session_manager: Optional["SessionManager"] = None,
    ):
        self.config = config
        self.faiss_manager = faiss_manager
        self.session_manager = session_manager

        self.host = str(config.get("host", "127.0.0.1"))
        self.port = int(config.get("port", 8080))
        self.session_timeout = max(60, int(config.get("session_timeout", 3600)))
        self._access_password = str(config.get("access_password", "")).strip()

        # Token 管理
        self._tokens: Dict[str, Dict[str, float]] = {}
        self._token_lock = asyncio.Lock()

        # 请求频率限制（防止暴力破解）
        self._failed_attempts: Dict[str, List[float]] = {}
        self._attempt_lock = asyncio.Lock()

        # 核爆功能
        self._pending_nuke: Optional[Dict[str, Any]] = None
        self._nuke_task: Optional[asyncio.Task] = None
        self._nuke_lock = asyncio.Lock()

        # 服务器状态
        self.memory_storage: Optional[MemoryStorage] = None
        self._storage_prepared = False
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._cleanup_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # 启动和停止
    # ------------------------------------------------------------------

    async def start(self):
        """启动 WebUI 服务"""
        if self._runner and not self._runner.closed:
            logger.warning("WebUI 服务已经在运行")
            return

        await self._prepare_storage()

        # 创建 aiohttp 应用
        self._app = web.Application()
        self._setup_routes()
        self._setup_middlewares()

        # 启动服务器
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

        # 启动定期清理任务
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

        logger.info(f"WebUI 已启动: http://{self.host}:{self.port}")

    async def stop(self):
        """停止 WebUI 服务"""
        if not self._runner:
            logger.debug("WebUI 服务未运行，无需停止")
            return

        logger.info("开始停止 WebUI 服务...")

        # 1. 取消清理任务
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # 2. 取消核爆任务
        if self._nuke_task and not self._nuke_task.done():
            self._nuke_task.cancel()
            try:
                await self._nuke_task
            except asyncio.CancelledError:
                pass

        # 3. 关闭服务器（aiohttp 的关闭非常干净）
        if self._site:
            await self._site.stop()
        
        if self._runner:
            await self._runner.cleanup()

        # 4. 清理状态
        self._app = None
        self._runner = None
        self._site = None
        self._cleanup_task = None
        self._nuke_task = None
        self._pending_nuke = None

        logger.info("WebUI 已停止")

    # ------------------------------------------------------------------
    # 路由设置
    # ------------------------------------------------------------------

    def _setup_routes(self):
        """设置路由"""
        # 静态文件
        static_dir = Path(__file__).resolve().parent.parent / "static"
        if static_dir.exists():
            self._app.router.add_static('/static', static_dir, name='static')

        # 首页
        self._app.router.add_get('/', self._serve_index)

        # API 路由
        self._app.router.add_post('/api/login', self._login)
        self._app.router.add_post('/api/logout', self._logout)
        self._app.router.add_get('/api/memories', self._list_memories)
        self._app.router.add_get('/api/memories/{memory_id}', self._memory_detail)
        self._app.router.add_delete('/api/memories', self._delete_memories)
        self._app.router.add_post('/api/memories/nuke', self._schedule_nuke)
        self._app.router.add_get('/api/memories/nuke', self._get_nuke_status)
        self._app.router.add_delete('/api/memories/nuke/{operation_id}', self._cancel_nuke)
        self._app.router.add_get('/api/stats', self._stats)
        self._app.router.add_get('/api/health', self._health)

    def _setup_middlewares(self):
        """设置中间件"""
        @web.middleware
        async def cors_middleware(request: web.Request, handler):
            """CORS 中间件"""
            # 处理 OPTIONS 预检请求
            if request.method == 'OPTIONS':
                response = web.Response()
            else:
                response = await handler(request)

            # 添加 CORS 头
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, DELETE, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Auth-Token'
            response.headers['Access-Control-Allow-Credentials'] = 'true'
            return response

        self._app.middlewares.append(cors_middleware)

    # ------------------------------------------------------------------
    # 路由处理函数
    # ------------------------------------------------------------------

    async def _serve_index(self, request: web.Request):
        """首页"""
        index_path = Path(__file__).resolve().parent.parent / "static" / "index.html"
        if not index_path.exists():
            raise web.HTTPNotFound(text="前端文件缺失")
        return web.Response(text=index_path.read_text(encoding='utf-8'), content_type='text/html')

    async def _login(self, request: web.Request):
        """登录"""
        try:
            data = await request.json()
        except Exception:
            raise web.HTTPBadRequest(text=json.dumps({"detail": "无效的 JSON"}))

        password = str(data.get("password", "")).strip()
        if not password:
            raise web.HTTPBadRequest(text=json.dumps({"detail": "密码不能为空"}))

        # 检查频率限制
        client_ip = request.remote or "unknown"
        if not await self._check_rate_limit(client_ip):
            raise web.HTTPTooManyRequests(text=json.dumps({"detail": "尝试次数过多，请5分钟后再试"}))

        # 验证密码
        if password != self._access_password:
            await self._record_failed_attempt(client_ip)
            await asyncio.sleep(1.0)
            raise web.HTTPUnauthorized(text=json.dumps({"detail": "认证失败"}))

        # 生成 token
        token = secrets.token_urlsafe(32)
        now = time.time()
        async with self._token_lock:
            await self._cleanup_tokens_locked()
            self._tokens[token] = {
                "created_at": now,
                "last_active": now,
                "max_lifetime": 86400  # 24小时
            }

        return web.json_response({
            "token": token,
            "expires_in": self.session_timeout
        })

    async def _logout(self, request: web.Request):
        """登出"""
        token = await self._extract_token(request)
        if token:
            async with self._token_lock:
                self._tokens.pop(token, None)
        return web.json_response({"detail": "已退出登录"})

    async def _list_memories(self, request: web.Request):
        """获取记忆列表"""
        await self._require_auth(request)

        # 解析查询参数
        keyword = request.query.get("keyword", "").strip()
        status_filter = request.query.get("status", "all").strip() or "all"
        load_all = request.query.get("all", "false").lower() == "true"

        if load_all:
            page = 1
            page_size = 0
            offset = 0
        else:
            page = max(1, int(request.query.get("page", 1)))
            page_size = request.query.get("page_size")
            page_size = min(200, max(1, int(page_size))) if page_size else 50
            offset = (page - 1) * page_size

        try:
            total, items = await self._fetch_memories(
                page=page,
                page_size=page_size,
                offset=offset,
                status_filter=status_filter,
                keyword=keyword,
                load_all=load_all,
            )
        except Exception as exc:
            logger.error(f"获取记忆列表失败: {exc}", exc_info=True)
            raise web.HTTPInternalServerError(text=json.dumps({"detail": "读取记忆失败"}))

        has_more = False if load_all else offset + len(items) < total
        effective_page_size = page_size if page_size else len(items)

        return web.json_response({
            "items": items,
            "page": page,
            "page_size": effective_page_size,
            "total": total,
            "has_more": has_more,
        })

    async def _memory_detail(self, request: web.Request):
        """获取记忆详情"""
        await self._require_auth(request)

        memory_id = request.match_info['memory_id']
        detail = await self._get_memory_detail(memory_id)
        if not detail:
            raise web.HTTPNotFound(text=json.dumps({"detail": "未找到记忆记录"}))
        return web.json_response(detail)

    async def _delete_memories(self, request: web.Request):
        """删除记忆"""
        await self._require_auth(request)

        try:
            data = await request.json()
        except Exception:
            raise web.HTTPBadRequest(text=json.dumps({"detail": "无效的 JSON"}))

        doc_ids = data.get("doc_ids") or data.get("ids") or []
        memory_ids = data.get("memory_ids") or []

        if not doc_ids and not memory_ids:
            raise web.HTTPBadRequest(text=json.dumps({"detail": "需要提供待删除的记忆ID列表"}))

        deleted_docs = 0
        deleted_memories = 0

        if doc_ids:
            try:
                doc_ids_int = [int(x) for x in doc_ids]
                await self.faiss_manager.delete_memories(doc_ids_int)
                deleted_docs = len(doc_ids_int)
            except Exception as exc:
                logger.error(f"删除 Faiss 记忆失败: {exc}", exc_info=True)
                raise web.HTTPInternalServerError(text=json.dumps({"detail": "向量记忆删除失败"}))

        if memory_ids and self.memory_storage:
            try:
                ids = [str(x) for x in memory_ids]
                await self.memory_storage.delete_memories_by_memory_ids(ids)
                deleted_memories = len(ids)
            except Exception as exc:
                logger.error(f"删除结构化记忆失败: {exc}", exc_info=True)
                raise web.HTTPInternalServerError(text=json.dumps({"detail": "结构化记忆删除失败"}))

        return web.json_response({
            "deleted_doc_count": deleted_docs,
            "deleted_memory_count": deleted_memories,
        })

    async def _schedule_nuke(self, request: web.Request):
        """调度核爆任务"""
        await self._require_auth(request)

        delay = 30
        try:
            data = await request.json()
            if "delay" in data:
                delay = int(data["delay"])
        except Exception:
            pass

        result = await self._do_schedule_nuke(delay)
        return web.json_response(result)

    async def _get_nuke_status(self, request: web.Request):
        """获取核爆状态"""
        await self._require_auth(request)

        async with self._nuke_lock:
            pending = self._pending_nuke
            if not pending or pending.get("status") != "scheduled":
                return web.json_response({"pending": False})
            snapshot = dict(pending)
        
        return web.json_response(self._serialize_nuke_status(snapshot))

    async def _cancel_nuke(self, request: web.Request):
        """取消核爆任务"""
        await self._require_auth(request)

        operation_id = request.match_info['operation_id']
        cancelled = await self._do_cancel_nuke(operation_id)
        if not cancelled:
            raise web.HTTPNotFound(text=json.dumps({"detail": "当前没有匹配的核爆任务"}))
        
        return web.json_response({
            "detail": "已取消核爆任务",
            "operation_id": operation_id
        })

    async def _stats(self, request: web.Request):
        """统计信息"""
        await self._require_auth(request)

        total, status_counts = await self._gather_statistics()
        active_sessions = (
            self.session_manager.get_session_count()
            if self.session_manager
            else 0
        )

        return web.json_response({
            "total_memories": total,
            "status_breakdown": status_counts,
            "active_sessions": active_sessions,
            "session_timeout": self.session_timeout,
        })

    async def _health(self, request: web.Request):
        """健康检查"""
        return web.json_response({"status": "ok"})

    # ------------------------------------------------------------------
    # 认证辅助函数
    # ------------------------------------------------------------------

    async def _extract_token(self, request: web.Request) -> Optional[str]:
        """从请求中提取 token"""
        # 从 Authorization 头
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:].strip()
        
        # 从 Cookie
        cookie_token = request.cookies.get("auth_token")
        if cookie_token:
            return cookie_token.strip()
        
        # 从自定义头
        custom_header = request.headers.get("X-Auth-Token", "")
        return custom_header.strip() if custom_header else None

    async def _require_auth(self, request: web.Request):
        """验证请求是否已认证"""
        token = await self._extract_token(request)
        if not token:
            raise web.HTTPUnauthorized(text=json.dumps({"detail": "未授权"}))

        async with self._token_lock:
            await self._cleanup_tokens_locked()
            token_data = self._tokens.get(token)

            if not token_data:
                raise web.HTTPUnauthorized(text=json.dumps({"detail": "会话已失效"}))

            now = time.time()

            # 检查绝对过期时间
            if now - token_data["created_at"] > token_data["max_lifetime"]:
                self._tokens.pop(token, None)
                raise web.HTTPUnauthorized(text=json.dumps({"detail": "会话已达最大时长"}))

            # 检查会话超时
            if now - token_data["last_active"] > self.session_timeout:
                self._tokens.pop(token, None)
                raise web.HTTPUnauthorized(text=json.dumps({"detail": "会话已过期"}))

            # 更新最后活动时间
            token_data["last_active"] = now

    async def _cleanup_tokens_locked(self):
        """清理过期 token（需要已持有锁）"""
        now = time.time()
        expired = []

        for token, token_data in self._tokens.items():
            if (now - token_data["created_at"] > token_data["max_lifetime"] or
                now - token_data["last_active"] > self.session_timeout):
                expired.append(token)

        for token in expired:
            self._tokens.pop(token, None)

    async def _check_rate_limit(self, client_ip: str) -> bool:
        """检查请求频率限制"""
        async with self._attempt_lock:
            await self._cleanup_failed_attempts_locked()
            attempts = self._failed_attempts.get(client_ip, [])
            recent = [t for t in attempts if time.time() - t < 300]

            if len(recent) >= 5:
                return False
            return True

    async def _record_failed_attempt(self, client_ip: str):
        """记录失败的登录尝试"""
        async with self._attempt_lock:
            if client_ip not in self._failed_attempts:
                self._failed_attempts[client_ip] = []
            self._failed_attempts[client_ip].append(time.time())

    async def _cleanup_failed_attempts_locked(self):
        """清理过期的失败尝试记录"""
        now = time.time()
        expired_ips = []
        for ip, attempts in self._failed_attempts.items():
            recent = [t for t in attempts if now - t < 300]
            if recent:
                self._failed_attempts[ip] = recent
            else:
                expired_ips.append(ip)

        for ip in expired_ips:
            self._failed_attempts.pop(ip, None)

    async def _periodic_cleanup(self):
        """定期清理任务"""
        while True:
            try:
                await asyncio.sleep(300)  # 每5分钟
                async with self._token_lock:
                    await self._cleanup_tokens_locked()
                async with self._attempt_lock:
                    await self._cleanup_failed_attempts_locked()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"定期清理任务出错: {e}")

    # ------------------------------------------------------------------
    # 存储和数据处理
    # ------------------------------------------------------------------

    async def _prepare_storage(self):
        """初始化存储"""
        if self._storage_prepared:
            return

        connection = None
        try:
            doc_storage = getattr(self.faiss_manager.db, "document_storage", None)
            connection = getattr(doc_storage, "connection", None)
        except Exception as exc:
            logger.debug(f"获取文档存储连接失败: {exc}")

        if connection:
            try:
                storage = MemoryStorage(connection)
                await storage.initialize_schema()
                self.memory_storage = storage
                logger.info("WebUI 已接入插件自定义的记忆存储（SQLite）")
            except Exception as exc:
                logger.warning(f"初始化 MemoryStorage 失败: {exc}")
                self.memory_storage = None
        else:
            logger.debug("未获取到 MemoryStorage 连接")

        self._storage_prepared = True

    async def _fetch_memories(
        self,
        page: int,
        page_size: int,
        offset: int,
        status_filter: str,
        keyword: str,
        load_all: bool,
    ) -> tuple[int, list]:
        """获取记忆列表"""
        try:
            total, records = await self._query_faiss_memories(
                offset=offset,
                page_size=page_size,
                status_filter=status_filter,
                keyword=keyword,
                load_all=load_all,
            )
        except Exception as exc:
            logger.error(f"查询记忆失败: {exc}", exc_info=True)
            total, records = await self._fetch_memories_fallback(
                offset=offset,
                page_size=page_size,
                status_filter=status_filter,
                keyword=keyword,
                load_all=load_all,
            )

        items = [self._format_memory(record, source="faiss") for record in records]
        return total, items

    async def _query_faiss_memories(
        self,
        offset: int,
        page_size: int,
        status_filter: str,
        keyword: str,
        load_all: bool,
    ) -> tuple[int, List[Dict[str, Any]]]:
        """从 Faiss 存储查询记忆"""
        doc_storage = getattr(self.faiss_manager.db, "document_storage", None)
        connection = getattr(doc_storage, "connection", None)
        if connection is None:
            raise RuntimeError("Document storage connection unavailable")

        conditions: List[str] = []
        params: List[Any] = []

        status_value = (status_filter or "").strip().lower()
        if status_value and status_value != "all":
            conditions.append("LOWER(COALESCE(json_extract(metadata, '$.status'), 'active')) = ?")
            params.append(status_value)

        keyword_value = (keyword or "").strip()
        if keyword_value:
            keyword_param = f"%{keyword_value.lower()}%"
            conditions.append("("
                "LOWER(text) LIKE ? OR "
                "LOWER(COALESCE(json_extract(metadata, '$.memory_content'), '')) LIKE ? OR "
                "LOWER(COALESCE(json_extract(metadata, '$.memory_id'), '')) LIKE ?"
                ")")
            params.extend([keyword_param, keyword_param, keyword_param])

        where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""

        count_sql = f"SELECT COUNT(*) FROM documents{where_clause}"
        async with connection.execute(count_sql, params) as cursor:
            row = await cursor.fetchone()
        total = int(row[0]) if row and row[0] is not None else 0

        query_sql = (
            "SELECT id, text, metadata FROM documents"
            f"{where_clause} ORDER BY id DESC"
        )
        query_params = list(params)
        if not load_all and page_size > 0:
            query_sql += " LIMIT ? OFFSET ?"
            query_params.extend([page_size, offset])

        async with connection.execute(query_sql, query_params) as cursor:
            rows = await cursor.fetchall()

        records: List[Dict[str, Any]] = []
        for row in rows:
            metadata_raw = row[2]
            if isinstance(metadata_raw, str):
                try:
                    metadata = json.loads(metadata_raw)
                except json.JSONDecodeError:
                    metadata = {}
            else:
                metadata = metadata_raw or {}

            records.append({
                "id": row[0],
                "content": row[1],
                "metadata": metadata,
            })

        return total, records

    async def _fetch_memories_fallback(
        self,
        offset: int,
        page_size: int,
        status_filter: str,
        keyword: str,
        load_all: bool,
    ) -> tuple[int, List[Dict[str, Any]]]:
        """回退方案：从 Faiss 接口获取记忆"""
        total_available = await self.faiss_manager.count_total_memories()
        fetch_size = max(total_available, page_size if page_size else 0, 1)

        records = await self.faiss_manager.get_memories_paginated(
            page_size=fetch_size, offset=0
        )

        filtered_records = self._filter_records(records, status_filter, keyword)
        total_filtered = len(filtered_records)

        if load_all:
            return total_filtered, filtered_records

        start = max(0, offset)
        end = start + page_size if page_size else total_filtered
        return total_filtered, filtered_records[start:end]

    def _filter_records(
        self,
        records: List[Dict[str, Any]],
        status_filter: str,
        keyword: str
    ) -> List[Dict[str, Any]]:
        """在内存中过滤记录"""
        filtered = []

        for record in records:
            metadata = record.get("metadata", {})

            # 状态过滤
            if status_filter and status_filter != "all":
                record_status = metadata.get("status", "active")
                if record_status != status_filter:
                    continue

            # 关键词过滤
            if keyword:
                content = record.get("content", "")
                memory_content = metadata.get("memory_content", "")
                keyword_lower = keyword.lower()

                if (keyword_lower not in content.lower() and
                    keyword_lower not in memory_content.lower()):
                    continue

            filtered.append(record)

        return filtered

    async def _get_memory_detail(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """获取单个记忆详情"""
        try:
            doc_id = int(memory_id)
        except ValueError:
            doc_id = None

        try:
            if doc_id is not None:
                docs = await self.faiss_manager.db.document_storage.get_documents(
                    ids=[doc_id]
                )
            else:
                all_docs = await self.faiss_manager.get_memories_paginated(
                    page_size=10000, offset=0
                )
                docs = [
                    doc for doc in all_docs
                    if doc.get("metadata", {}).get("memory_id") == memory_id
                ]

            if not docs:
                return None

            return self._format_memory(docs[0], source="faiss")

        except Exception as exc:
            logger.error(f"查询记忆详情失败: {exc}", exc_info=True)
            return None

    def _format_memory(self, raw: Dict[str, Any], source: str) -> Dict[str, Any]:
        """格式化记忆数据"""
        metadata = raw.get("metadata", {})

        summary = metadata.get("memory_content") or raw.get("content") or ""
        importance = metadata.get("importance")
        event_type = metadata.get("event_type")
        status = metadata.get("status", "active")
        created_at = metadata.get("create_time")
        last_access = metadata.get("last_access_time")

        return {
            "doc_id": raw.get("id"),
            "memory_id": metadata.get("memory_id"),
            "summary": summary,
            "memory_type": event_type,
            "importance": importance,
            "status": status,
            "created_at": self._format_timestamp(created_at),
            "last_access": self._format_timestamp(last_access),
            "source": "faiss",
            "metadata": metadata,
            "raw": {
                "content": raw.get("content"),
                "metadata": metadata,
            },
            "raw_json": json.dumps(metadata, ensure_ascii=False),
        }

    async def _gather_statistics(self) -> tuple[int, Dict[str, int]]:
        """统计记忆数量"""
        total = await self.faiss_manager.count_total_memories()
        counts = await self._collect_status_counts()
        return total, counts

    async def _collect_status_counts(self) -> Dict[str, int]:
        """统计不同状态的记忆数量"""
        counts: Dict[str, int] = {"active": 0, "archived": 0, "deleted": 0}
        try:
            conn = self.faiss_manager.db.document_storage.connection
            async with conn.execute(
                "SELECT json_extract(metadata, '$.status') AS status FROM documents"
            ) as cursor:
                rows = await cursor.fetchall()
            for row in rows:
                status_value = row[0] if row and row[0] else "active"
                counts[status_value] = counts.get(status_value, 0) + 1
        except Exception as exc:
            logger.error(f"统计记忆状态失败: {exc}", exc_info=True)
        return counts

    # ------------------------------------------------------------------
    # 核爆功能
    # ------------------------------------------------------------------

    async def _do_schedule_nuke(self, delay_seconds: int) -> Dict[str, Any]:
        """调度核爆任务"""
        delay = max(5, min(int(delay_seconds), 600))
        task_to_cancel: Optional[asyncio.Task] = None
        pending_snapshot: Dict[str, Any]

        async with self._nuke_lock:
            now = time.time()
            if self._pending_nuke and self._pending_nuke.get("status") == "scheduled":
                return self._serialize_nuke_status(self._pending_nuke, now, True)

            if self._nuke_task and not self._nuke_task.done():
                task_to_cancel = self._nuke_task

            operation_id = secrets.token_urlsafe(8)
            execute_at = now + delay
            pending = {
                "id": operation_id,
                "created_at": now,
                "execute_at": execute_at,
                "status": "scheduled",
            }
            self._pending_nuke = pending
            self._nuke_task = asyncio.create_task(self._run_nuke(operation_id, delay))
            pending_snapshot = dict(pending)

        if task_to_cancel:
            task_to_cancel.cancel()
            try:
                await task_to_cancel
            except asyncio.CancelledError:
                pass

        return self._serialize_nuke_status(pending_snapshot, time.time())

    async def _do_cancel_nuke(self, operation_id: str) -> bool:
        """取消核爆任务"""
        task: Optional[asyncio.Task] = None
        async with self._nuke_lock:
            if not self._pending_nuke or self._pending_nuke.get("id") != operation_id:
                return False
            task = self._nuke_task
            self._pending_nuke = None
            self._nuke_task = None

        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        return True

    async def _run_nuke(self, operation_id: str, delay: int):
        """运行核爆任务"""
        try:
            await asyncio.sleep(delay)
            await self._execute_nuke(operation_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"核爆任务失败: {exc}", exc_info=True)
            async with self._nuke_lock:
                if self._pending_nuke and self._pending_nuke.get("id") == operation_id:
                    self._pending_nuke = None
                self._nuke_task = None

    async def _execute_nuke(self, operation_id: str):
        """执行核爆（仅为视觉效果，不删除数据）"""
        async with self._nuke_lock:
            if not self._pending_nuke or self._pending_nuke.get("id") != operation_id:
                return
            self._pending_nuke["status"] = "running"

        logger.info("核爆视觉效果触发：这只是模拟，不会删除任何数据")

        try:
            async with self.faiss_manager.db.document_storage.connection.execute(
                "SELECT COUNT(*) FROM documents"
            ) as cursor:
                row = await cursor.fetchone()
                vector_deleted = row[0] if row else 0

            storage_deleted = 0
            if self.memory_storage:
                async with self.memory_storage.connection.execute(
                    "SELECT COUNT(*) FROM memories"
                ) as cursor:
                    row = await cursor.fetchone()
                    storage_deleted = row[0] if row else 0

            logger.info(
                f"核爆视觉效果完成：模拟清除 {vector_deleted} 条向量记录和 "
                f"{storage_deleted} 条结构化记录（实际数据完全未受影响）"
            )
        except Exception as exc:
            logger.error(f"核爆效果执行失败: {exc}", exc_info=True)
        finally:
            async with self._nuke_lock:
                if self._pending_nuke and self._pending_nuke.get("id") == operation_id:
                    self._pending_nuke = None
                self._nuke_task = None

    def _serialize_nuke_status(
        self,
        payload: Optional[Dict[str, Any]],
        now: Optional[float] = None,
        already_pending: bool = False,
    ) -> Dict[str, Any]:
        """序列化核爆状态"""
        if not payload:
            return {"pending": False}

        now = now or time.time()
        execute_at = float(payload.get("execute_at", now))
        seconds_left = max(0, int(round(execute_at - now)))
        if already_pending:
            detail = "A pending wipe is already counting down"
        else:
            detail = (
                f"Wipe executes in {seconds_left} seconds"
                if seconds_left
                else "Wipe executing now"
            )

        return {
            "pending": True,
            "operation_id": payload.get("id"),
            "execute_at": datetime.fromtimestamp(execute_at).isoformat(
                sep=" ", timespec="seconds"
            ),
            "seconds_left": seconds_left,
            "detail": detail,
            "already_pending": already_pending,
        }

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------

    @staticmethod
    def _format_timestamp(value: Any) -> Optional[str]:
        """格式化时间戳"""
        if not value:
            return None
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value).isoformat(sep=" ", timespec="seconds")
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat(
                    sep=" ", timespec="seconds"
                )
            except ValueError:
                return value
        return str(value)
