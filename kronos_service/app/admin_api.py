"""/api/admin endpoints: login, monitoring overview, keys, config, models, logs, selftest."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import psutil
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from . import __version__
from .auth import AdminAuth, generate_api_key, require_admin
from .db_models import ApiKey, RequestLog
from .model_manager import REGISTRY

logger = logging.getLogger("kronos.api.admin")

router = APIRouter(prefix="/api/admin", tags=["admin"])


# ---------------------------------------------------------------------- login
class LoginBody(BaseModel):
    username: str
    password: str


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@router.post("/login")
async def login(body: LoginBody, request: Request):
    state = request.app.state
    admin: AdminAuth = state.admin_auth
    lock_left = admin.is_locked(_client_ip(request), body.username)
    if lock_left:
        raise HTTPException(status_code=429, detail=f"尝试次数过多，请 {lock_left} 秒后重试")
    if not admin.check_credentials(_client_ip(request), body.username, body.password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return {"token": admin.issue_token(), "expires_in": admin.expire_hours * 3600, "username": body.username}


# ------------------------------------------------------------------- overview
@router.get("/overview")
async def overview(request: Request, _: dict = Depends(require_admin)):
    state = request.app.state
    now = datetime.now(timezone.utc)
    session = state.session_factory()
    try:
        hour_ago, day_ago = now - timedelta(hours=1), now - timedelta(days=1)

        def _window(start: datetime) -> dict[str, Any]:
            total = session.execute(
                select(func.count(RequestLog.id)).where(RequestLog.ts >= start)
            ).scalar_one()
            ok = session.execute(
                select(func.count(RequestLog.id)).where(
                    RequestLog.ts >= start, RequestLog.status_code < 400
                )
            ).scalar_one()
            err4 = session.execute(
                select(func.count(RequestLog.id)).where(
                    RequestLog.ts >= start, RequestLog.status_code.between(400, 499)
                )
            ).scalar_one()
            err5 = session.execute(
                select(func.count(RequestLog.id)).where(
                    RequestLog.ts >= start, RequestLog.status_code >= 500
                )
            ).scalar_one()
            avg = session.execute(
                select(func.avg(RequestLog.latency_ms)).where(
                    RequestLog.ts >= start, RequestLog.status_code < 400
                )
            ).scalar_one()
            rows = session.execute(
                select(RequestLog.latency_ms).where(
                    RequestLog.ts >= start, RequestLog.status_code < 400
                ).order_by(RequestLog.latency_ms)
            ).scalars().all()
            p50 = rows[int(len(rows) * 0.50)] if rows else 0
            p95 = rows[min(int(len(rows) * 0.95), len(rows) - 1)] if rows else 0
            return {
                "total": int(total or 0),
                "success": int(ok or 0),
                "client_errors": int(err4 or 0),
                "server_errors": int(err5 or 0),
                "success_rate": round((ok / total) * 100, 2) if total else None,
                "latency_avg_ms": int(avg or 0),
                "latency_p50_ms": int(p50),
                "latency_p95_ms": int(p95),
            }

        # hourly series for charts (last 24h, UTC)
        since = now - timedelta(hours=24)
        logs = session.execute(
            select(RequestLog.ts, RequestLog.status_code, RequestLog.latency_ms, RequestLog.model, RequestLog.symbol, RequestLog.api_key_name).where(RequestLog.ts >= since)
        ).all()

        buckets: dict[str, dict[str, list[int]]] = {}
        for ts, code, latency, model, symbol, key_name in logs:
            hour = ts.replace(minute=0, second=0, microsecond=0).isoformat()
            b = buckets.setdefault(hour, {"total": [], "ok": [], "err": []})
            b["total"].append(latency or 0)
            (b["ok"] if (code or 0) < 400 else b["err"]).append(latency or 0)
        series = [
            {
                "hour": h,
                "count": len(v["total"]),
                "ok": len(v["ok"]),
                "error": len(v["err"]),
                "latency_avg_ms": int(sum(v["total"]) / len(v["total"])) if v["total"] else 0,
            }
            for h, v in sorted(buckets.items())
        ]

        def _top(field, n=8):
            rows_ = session.execute(
                select(field, func.count(RequestLog.id))
                .where(RequestLog.ts >= since)
                .group_by(field)
                .order_by(func.count(RequestLog.id).desc())
                .limit(n)
            ).all()
            return [{"name": r[0] or "-", "count": int(r[1])} for r in rows_]

        requests_1h = _window(hour_ago)
        requests_24h = _window(day_ago)
    finally:
        session.close()

    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(str(state.settings.instance_dir)) if state.settings.instance_dir.exists() else None
    started = state.started_at
    return {
        "service": {
            "version": __version__,
            "started_at": datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
            "uptime_s": int(time.time() - started),
            "env": state.settings.env,
        },
        "queue": state.queue.metrics(),
        "models": state.model_manager.status(),
        "gpu": state.model_manager.gpu_info(),
        "requests_1h": requests_1h,
        "requests_24h": requests_24h,
        "hourly_series": series,
        "top_keys": _top(RequestLog.api_key_name),
        "top_symbols": _top(RequestLog.symbol),
        "top_errors": _top(RequestLog.error, n=5),
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=0.3),
            "mem_total_gb": round(vm.total / 1024**3, 1),
            "mem_used_gb": round(vm.used / 1024**3, 1),
            "mem_percent": vm.percent,
            "disk_used_gb": round(disk.used / 1024**3, 1) if disk else None,
            "disk_total_gb": round(disk.total / 1024**3, 1) if disk else None,
        },
        "config": {c["key"]: c["value"] for c in state.cfg.snapshot()},
    }


# ----------------------------------------------------------------------- keys
class KeyCreateBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    note: str = ""
    rate_limit_per_min: int = Field(0, ge=0, le=10000)
    daily_quota: int = Field(0, ge=0, le=10_000_000)
    expires_days: int | None = Field(None, ge=1, le=3650, description="有效天数, 空=永久")


class KeyUpdateBody(BaseModel):
    name: str | None = None
    note: str | None = None
    enabled: bool | None = None
    rate_limit_per_min: int | None = Field(None, ge=0, le=10000)
    daily_quota: int | None = Field(None, ge=0, le=10_000_000)
    expires_days: int | None = Field(None, ge=1, le=3650)


def _key_row(row: ApiKey) -> dict[str, Any]:
    return {
        "id": row.id,
        "key_prefix": row.key_prefix,
        "name": row.name,
        "note": row.note,
        "enabled": bool(row.enabled and row.revoked_at is None),
        "rate_limit_per_min": row.rate_limit_per_min,
        "daily_quota": row.daily_quota,
        "expires_at": row.expires_at.isoformat() if row.expires_at else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "last_used_at": row.last_used_at.isoformat() if row.last_used_at else None,
        "total_calls": row.total_calls,
    }


@router.get("/keys")
async def list_keys(request: Request, _: dict = Depends(require_admin)):
    state = request.app.state
    session = state.session_factory()
    try:
        rows = session.query(ApiKey).order_by(ApiKey.id.desc()).all()
        return {"keys": [_key_row(r) for r in rows]}
    finally:
        session.close()


@router.post("/keys")
async def create_key(body: KeyCreateBody, request: Request, _: dict = Depends(require_admin)):
    state = request.app.state
    raw, key_hash, prefix = generate_api_key()
    expires = (
        datetime.now(timezone.utc) + timedelta(days=body.expires_days) if body.expires_days else None
    )
    session = state.session_factory()
    try:
        row = ApiKey(
            key_hash=key_hash,
            key_prefix=prefix,
            name=body.name,
            note=body.note,
            enabled=True,
            rate_limit_per_min=body.rate_limit_per_min,
            daily_quota=body.daily_quota,
            expires_at=expires,
        )
        session.add(row)
        session.commit()
        state.key_cache.refresh_key(row.id)
        logger.info("签发 API 密钥: %s (%s)", prefix, body.name)
        return {"key": _key_row(row), "plaintext": raw}  # shown exactly once
    finally:
        session.close()


@router.patch("/keys/{key_id}")
async def update_key(key_id: int, body: KeyUpdateBody, request: Request, _: dict = Depends(require_admin)):
    state = request.app.state
    session = state.session_factory()
    try:
        row = session.get(ApiKey, key_id)
        if row is None:
            raise HTTPException(status_code=404, detail="密钥不存在")
        if body.name is not None:
            row.name = body.name
        if body.note is not None:
            row.note = body.note
        if body.enabled is not None:
            row.enabled = body.enabled
            if body.enabled:
                row.revoked_at = None
        if body.rate_limit_per_min is not None:
            row.rate_limit_per_min = body.rate_limit_per_min
        if body.daily_quota is not None:
            row.daily_quota = body.daily_quota
        if body.expires_days is not None:
            row.expires_at = datetime.now(timezone.utc) + timedelta(days=body.expires_days)
        session.commit()
        state.key_cache.refresh_key(row.id)
        return {"key": _key_row(row)}
    finally:
        session.close()


@router.delete("/keys/{key_id}")
async def revoke_key(key_id: int, request: Request, _: dict = Depends(require_admin)):
    state = request.app.state
    session = state.session_factory()
    try:
        row = session.get(ApiKey, key_id)
        if row is None:
            raise HTTPException(status_code=404, detail="密钥不存在")
        row.enabled = False
        row.revoked_at = datetime.now(timezone.utc)
        session.commit()
        state.key_cache.refresh_key(row.id)
        state.rate_limiter.reset(key_id)
        logger.info("吊销 API 密钥: %s (%s)", row.key_prefix, row.name)
        return {"key": _key_row(row)}
    finally:
        session.close()


@router.get("/keys/{key_id}/usage")
async def key_usage(key_id: int, request: Request, limit: int = 50, _: dict = Depends(require_admin)):
    state = request.app.state
    session = state.session_factory()
    try:
        row = session.get(ApiKey, key_id)
        if row is None:
            raise HTTPException(status_code=404, detail="密钥不存在")
        day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        today_count = session.execute(
            select(func.count(RequestLog.id)).where(
                RequestLog.api_key_id == key_id, RequestLog.ts >= day_start
            )
        ).scalar_one()
        logs = session.execute(
            select(RequestLog)
            .where(RequestLog.api_key_id == key_id)
            .order_by(RequestLog.ts.desc())
            .limit(min(limit, 200))
        ).scalars().all()
        return {
            "key": _key_row(row),
            "today_calls": int(today_count or 0),
            "recent_logs": [
                {
                    "ts": l.ts.isoformat(),
                    "endpoint": l.endpoint,
                    "status_code": l.status_code,
                    "latency_ms": l.latency_ms,
                    "model": l.model,
                    "symbol": l.symbol,
                    "pred_len": l.pred_len,
                    "error": l.error,
                }
                for l in logs
            ],
        }
    finally:
        session.close()


# --------------------------------------------------------------------- config
@router.get("/config")
async def get_config(request: Request, _: dict = Depends(require_admin)):
    return {"config": request.app.state.cfg.snapshot()}


@router.put("/config")
async def put_config(values: dict[str, Any], request: Request, _: dict = Depends(require_admin)):
    state = request.app.state
    try:
        applied, restart_keys = state.cfg.set_many(values)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=json.loads(str(e)))
    state.writer.set_retention(
        state.cfg.get_int("log_retention_days"), state.cfg.get_int("task_retention_hours")
    )
    return {"applied": applied, "restart_required": restart_keys}


# --------------------------------------------------------------------- models
@router.get("/models")
async def models_status(request: Request, _: dict = Depends(require_admin)):
    state = request.app.state
    return {
        "models": state.model_manager.status(),
        "gpu": state.model_manager.gpu_info(),
        "default_model": state.cfg.get("default_model"),
    }


@router.post("/models/{key}/load")
async def model_load(key: str, request: Request, _: dict = Depends(require_admin)):
    state = request.app.state
    if key not in REGISTRY:
        raise HTTPException(status_code=404, detail=f"未知模型, 可选: {list(REGISTRY)}")

    def _load():
        try:
            state.model_manager.load(key)
        except Exception:  # noqa: BLE001
            logger.exception("后台加载模型 %s 失败", key)

    asyncio.get_running_loop().run_in_executor(None, _load)
    return {"status": "loading", "model": key, "hint": "加载需数秒至数分钟, 请稍后刷新状态"}


@router.post("/models/{key}/unload")
async def model_unload(key: str, request: Request, _: dict = Depends(require_admin)):
    state = request.app.state
    if key not in REGISTRY:
        raise HTTPException(status_code=404, detail="未知模型")

    def _unload():
        try:
            state.model_manager.unload(key)
        except Exception:  # noqa: BLE001
            logger.exception("卸载模型 %s 失败", key)

    asyncio.get_running_loop().run_in_executor(None, _unload)
    return {"status": "unloading", "model": key}


# ----------------------------------------------------------------------- logs
@router.get("/logs")
async def list_logs(
    request: Request,
    limit: int = 50,
    status: str | None = None,
    key_id: int | None = None,
    symbol: str | None = None,
    _: dict = Depends(require_admin),
):
    state = request.app.state
    session = state.session_factory()
    try:
        q = select(RequestLog).order_by(RequestLog.ts.desc()).limit(min(limit, 500))
        if status == "error":
            q = q.where(RequestLog.status_code >= 400)
        elif status and status.isdigit():
            q = q.where(RequestLog.status_code == int(status))
        if key_id:
            q = q.where(RequestLog.api_key_id == key_id)
        if symbol:
            q = q.where(RequestLog.symbol == symbol)
        rows = session.execute(q).scalars().all()
        return {
            "logs": [
                {
                    "ts": r.ts.isoformat(),
                    "api_key_name": r.api_key_name,
                    "endpoint": r.endpoint,
                    "status_code": r.status_code,
                    "latency_ms": r.latency_ms,
                    "queue_ms": r.queue_ms,
                    "inference_ms": r.inference_ms,
                    "model": r.model,
                    "symbol": r.symbol,
                    "pred_len": r.pred_len,
                    "task_id": r.task_id,
                    "client_ip": r.client_ip,
                    "error": r.error,
                }
                for r in rows
            ]
        }
    finally:
        session.close()


# ------------------------------------------------------------------- selftest
@router.post("/selftest")
async def selftest(request: Request, _: dict = Depends(require_admin)):
    from .selftest import run_all

    state = request.app.state
    smoke_model = state.cfg.get("default_model")
    t0 = time.perf_counter()
    checks = await asyncio.get_running_loop().run_in_executor(None, run_all, state, smoke_model)
    passed = sum(1 for c in checks if c["passed"])
    return {
        "passed": passed == len(checks),
        "summary": f"{passed}/{len(checks)} 项通过",
        "total_ms": int((time.perf_counter() - t0) * 1000),
        "checks": checks,
        "ran_at": datetime.now(timezone.utc).isoformat(),
    }
