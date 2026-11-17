"""FastAPI application factory and lifespan wiring."""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .admin_api import router as admin_router
from .auth import AdminAuth, DailyQuota, KeyCache, RateLimiter
from .db import init_db, make_engine, make_session_factory
from .db_writer import DbWriter
from .logging_setup import setup_logging
from .model_manager import ModelManager
from .predict_api import router as predict_router
from .runtime_config import RuntimeConfig
from .settings import Settings, get_settings
from .task_queue import InferenceQueue

logger = logging.getLogger("kronos.main")


def _ensure_repo_root_on_path() -> None:
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_repo_root_on_path()


def create_app(settings: Settings | None = None, preload: list[str] | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.instance_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(settings.log_dir, settings.log_level)
    for w in settings.warnings():
        logger.warning("%s", w)

    engine = make_engine(settings.db_path)
    init_db(engine)
    session_factory = make_session_factory(engine)

    cfg = RuntimeConfig(session_factory)
    overrides = {
        "max_concurrent_inferences": settings.max_concurrent_inferences,
        "max_queue_size": settings.max_queue_size,
        "sync_timeout_seconds": settings.sync_timeout_seconds,
        "log_retention_days": settings.log_retention_days,
        "task_retention_hours": settings.task_retention_hours,
    }
    if preload is not None:
        overrides["preload_models"] = preload
    cfg.load(overrides=overrides)

    writer = DbWriter(
        session_factory,
        retention={
            "request_log_days": cfg.get_int("log_retention_days"),
            "task_hours": cfg.get_int("task_retention_hours"),
        },
    )
    model_manager = ModelManager(settings.models_dir, device=settings.device)
    queue = InferenceQueue(model_manager, writer, cfg)
    admin_auth = AdminAuth(
        settings.admin_username, settings.admin_password, settings.jwt_secret, settings.jwt_expire_hours
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        writer.start()
        key_cache = KeyCache(session_factory)
        key_cache.reload()
        queue.start(
            n_workers=cfg.get_int("max_concurrent_inferences"),
            max_queue_size=cfg.get_int("max_queue_size"),
        )
        app.state.key_cache = key_cache
        app.state.rate_limiter = RateLimiter()
        app.state.daily_quota = DailyQuota(session_factory)

        preload: list[str] = list(cfg.get("preload_models") or [])
        if preload:
            logger.info("预加载模型: %s", preload)

            def _preload():
                for key in preload:
                    try:
                        model_manager.load(key)
                    except Exception:  # noqa: BLE001
                        logger.exception("预加载模型 %s 失败（服务仍启动）", key)

            await asyncio.get_running_loop().run_in_executor(None, _preload)
        logger.info("Kronos 预测服务 v%s 已就绪 (device=%s)", __version__, model_manager.resolve_device())
        yield
        logger.info("正在停止: 拒绝新请求并等待在跑推理收尾...")
        queue.shutdown(timeout=max(60, cfg.get_int("sync_timeout_seconds")))
        writer.stop()
        engine.dispose()

    app = FastAPI(
        title="Kronos 预测服务",
        version=__version__,
        description="Kronos K线预测模型 REST API（股票/指数等任意品种 K 线预测）",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    app.state.settings = settings
    app.state.session_factory = session_factory
    app.state.cfg = cfg
    app.state.writer = writer
    app.state.model_manager = model_manager
    app.state.queue = queue
    app.state.admin_auth = admin_auth
    app.state.started_at = time.time()

    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def body_size_and_auth_failure_log(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit():
            if int(content_length) > settings.max_request_body_mb * 1024 * 1024:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"请求体超过上限 {settings.max_request_body_mb}MB"},
                )
        response = await call_next(request)
        if request.url.path.startswith("/v1/") and response.status_code in (401, 403, 429):
            writer.log_request(
                endpoint=str(request.url.path),
                method=request.method,
                status_code=response.status_code,
                latency_ms=0,
                client_ip=request.headers.get("x-forwarded-for", "").split(",")[0].strip()
                or (request.client.host if request.client else "unknown"),
                error="auth/rate-limit rejected",
            )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors()[:10]})

    app.include_router(predict_router)
    app.include_router(admin_router)

    @app.get("/healthz", tags=["health"], summary="存活探针")
    async def healthz():
        return {
            "status": "ok",
            "version": __version__,
            "uptime_s": int(time.time() - app.state.started_at),
        }

    @app.get("/readyz", tags=["health"], summary="就绪探针(模型已加载且队列可用)")
    async def readyz():
        preload: list[str] = list(cfg.get("preload_models") or [])
        ready = queue.running() and (not preload or model_manager.any_loaded())
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "ready": ready,
                "queue_running": queue.running(),
                "models_loaded": [m["key"] for m in model_manager.status() if m["loaded"]],
            },
        )

    static_dir = Path(__file__).resolve().parent.parent / "static" / "admin"
    if static_dir.is_dir():
        app.mount("/admin", StaticFiles(directory=str(static_dir), html=True), name="admin")

    return app


app = create_app()
