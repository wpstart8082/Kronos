"""/v1 prediction API: models list, sync predict, async predict, task query."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .auth import KeyInfo, require_api_key
from .inference import DataValidationError, PredictParams, parse_candles
from .model_manager import REGISTRY
from .task_queue import QueueFullError
from .timestamps import TimestampError

logger = logging.getLogger("kronos.api.predict")

router = APIRouter(prefix="/v1", tags=["prediction"])


class Candle(BaseModel):
    timestamp: str = Field(..., description="K线时间戳, ISO8601, 如 2026-11-14 14:35:00")
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: float | None = None
    amount: float | None = None


class PredictRequest(BaseModel):
    model: str | None = Field(None, description="kronos-mini | kronos-small | kronos-base, 缺省用系统默认")
    symbol: str | None = Field(None, max_length=32, description="标的代码, 仅用于标识与统计, 如 300308 / 000300")
    data: list[Candle] = Field(..., min_length=2, description="历史K线, 需按时间升序(服务端会再次排序)")
    lookback: int | None = Field(None, ge=8, le=4096, description="参与预测的历史窗口长度")
    pred_len: int = Field(..., ge=1, le=480, description="预测步数")
    temperature: float = Field(1.0, gt=0, le=5)
    top_k: int = Field(0, ge=0, le=1024)
    top_p: float = Field(0.9, gt=0, le=1)
    sample_count: int = Field(1, ge=1, le=8, description="采样路径数, 结果取均值")
    future_timestamps: list[str] | None = Field(None, description="显式指定未来时间戳, 长度须等于 pred_len")


@router.get("/models")
async def list_models(request: Request, _: KeyInfo = Depends(require_api_key)):
    state = request.app.state
    return {"models": state.model_manager.status(), "default_model": state.cfg.get("default_model")}


@router.post("/predict")
async def predict_sync(body: PredictRequest, request: Request, key: KeyInfo = Depends(require_api_key)):
    import time as _time

    state = request.app.state
    t0 = _time.perf_counter()
    submitted = None
    try:
        submitted = await _submit(body, request, key, mode="sync")
        job, _ = submitted
        sync_timeout = state.cfg.get_int("sync_timeout_seconds")
        try:
            # shield: a sync timeout must NOT cancel the underlying job -
            # the client can still fetch the result via /v1/tasks/{task_id}
            wrapped = asyncio.shield(asyncio.wrap_future(job.future))
            result = await asyncio.wait_for(wrapped, timeout=sync_timeout)
        except asyncio.TimeoutError:
            logger.info("同步请求超时(%ss), task_id=%s 可通过任务接口查询", sync_timeout, job.task_id)
            _log(state, request, key, "/v1/predict", 408, body, job, total_ms=_time.perf_counter() - t0)
            raise HTTPException(
                status_code=408,
                detail={
                    "message": f"同步等待超过 {sync_timeout} 秒",
                    "task_id": job.task_id,
                    "status_url": f"/v1/tasks/{job.task_id}",
                },
            )
        _log(
            state, request, key, "/v1/predict", 200, body, job,
            total_ms=_time.perf_counter() - t0,
            queue_ms=result.queue_ms, inference_ms=result.inference_ms,
        )
        return _build_response(body, job, result)
    except HTTPException as e:
        if submitted is None:  # validation / queue-full failures
            _log(state, request, key, "/v1/predict", e.status_code, body, None, total_ms=_time.perf_counter() - t0, error=str(e.detail))
        raise


@router.post("/predict/async", status_code=202)
async def predict_async(body: PredictRequest, request: Request, key: KeyInfo = Depends(require_api_key)):
    import time as _time

    state = request.app.state
    t0 = _time.perf_counter()
    try:
        job, _ = await _submit(body, request, key, mode="async")
        _log(state, request, key, "/v1/predict/async", 202, body, job, total_ms=_time.perf_counter() - t0)
        return {"task_id": job.task_id, "status": "queued", "status_url": f"/v1/tasks/{job.task_id}"}
    except HTTPException as e:
        _log(state, request, key, "/v1/predict/async", e.status_code, body, None, total_ms=_time.perf_counter() - t0, error=str(e.detail))
        raise


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request, key: KeyInfo = Depends(require_api_key)):
    from .db_models import TaskRecord

    state = request.app.state
    session = state.session_factory()
    try:
        row = session.get(TaskRecord, task_id)
    finally:
        session.close()
    if row is None or row.api_key_id != key.id:
        raise HTTPException(status_code=404, detail="任务不存在")
    resp: dict[str, Any] = {
        "task_id": row.task_id,
        "status": row.status,
        "model": row.model,
        "symbol": row.symbol,
        "pred_len": row.pred_len,
        "queue_ms": row.queue_ms,
        "inference_ms": row.inference_ms,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
    }
    if row.status == "succeeded":
        import json as _json

        resp["result"] = _json.loads(row.result) if row.result else None
    if row.error:
        resp["error"] = row.error
    return resp


# ---------------------------------------------------------------------- shared
def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _log(state, request: Request, key: KeyInfo, endpoint: str, status_code: int,
         body: PredictRequest, job, total_ms: float, queue_ms: int = 0,
         inference_ms: int = 0, error: str | None = None):
    state.writer.log_request(
        api_key_id=key.id,
        api_key_name=key.name,
        endpoint=endpoint,
        method="POST",
        status_code=status_code,
        latency_ms=int(total_ms * 1000),
        queue_ms=queue_ms,
        inference_ms=inference_ms,
        model=(job.model_key if job else (body.model or state.cfg.get("default_model"))),
        symbol=body.symbol,
        pred_len=body.pred_len,
        lookback=body.lookback,
        sample_count=body.sample_count,
        task_id=job.task_id if job else None,
        client_ip=_client_ip(request),
        error=error[:500] if error else None,
    )


async def _submit(body: PredictRequest, request: Request, key: KeyInfo, mode: str):
    state = request.app.state
    cfg = state.cfg

    model_key = body.model or cfg.get("default_model")
    if model_key not in REGISTRY:
        raise HTTPException(status_code=422, detail=f"未知模型 {model_key}, 可选: {list(REGISTRY)}")
    if body.pred_len > cfg.get_int("max_pred_len"):
        raise HTTPException(status_code=422, detail=f"pred_len 超过当前配置上限 {cfg.get_int('max_pred_len')}")
    if body.sample_count > cfg.get_int("max_sample_count"):
        raise HTTPException(status_code=422, detail=f"sample_count 超过当前配置上限 {cfg.get_int('max_sample_count')}")
    if body.lookback and body.lookback > cfg.get_int("max_lookback"):
        raise HTTPException(status_code=422, detail=f"lookback 超过当前配置上限 {cfg.get_int('max_lookback')}")

    params = PredictParams(
        pred_len=body.pred_len,
        temperature=body.temperature,
        top_k=body.top_k,
        top_p=body.top_p,
        sample_count=body.sample_count,
    )
    try:
        prepared = parse_candles(
            data=[c.model_dump() for c in body.data],
            model_key=model_key,
            cfg=cfg,
            lookback_requested=body.lookback,
            pred_len=body.pred_len,
            future_timestamps=body.future_timestamps,
            symbol=body.symbol,
        )
    except (DataValidationError, TimestampError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    if not state.model_manager.is_loaded(model_key) and not state.queue.running():
        raise HTTPException(status_code=503, detail="服务尚未就绪，请稍后重试")

    try:
        job = state.queue.submit(
            model_key=model_key,
            params=params,
            prepared=prepared,
            api_key_id=key.id,
            api_key_name=key.name,
            mode=mode,
        )
    except QueueFullError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    state.daily_quota.record(key.id)
    state.writer.touch_key(key.id)
    return job, params


def _build_response(body: PredictRequest, job, result) -> dict[str, Any]:
    return {
        "task_id": job.task_id,
        "model": job.model_key,
        "symbol": body.symbol,
        "request": {
            "pred_len": job.params.pred_len,
            "lookback": job.prepared.lookback,
            "temperature": job.params.temperature,
            "top_k": job.params.top_k,
            "top_p": job.params.top_p,
            "sample_count": job.params.sample_count,
        },
        "input": {
            "from": str(job.prepared.input_from),
            "to": str(job.prepared.input_to),
            "points": int(job.prepared.lookback),
            "frequency": job.prepared.freq,
        },
        "timings": {
            "queue_ms": result.queue_ms,
            "inference_ms": result.inference_ms,
            "total_ms": result.queue_ms + result.inference_ms,
        },
        "predictions": result.predictions,
    }
