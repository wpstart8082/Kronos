"""Inference job queue: bounded queue + worker pool + optional dynamic batching.

Every job (sync or async) gets a TaskRecord via the DB writer, so a timed-out
sync request can still fetch its result from /v1/tasks/{task_id}.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

from .inference import (
    PredictParams,
    PreparedInput,
    check_prediction_sane,
    format_predictions,
    run_prediction,
    run_prediction_batch,
)

logger = logging.getLogger("kronos.queue")


def percentile(values: deque, p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(int(len(ordered) * p), len(ordered) - 1)
    return int(ordered[idx])


@dataclass
class Job:
    task_id: str
    model_key: str
    api_key_id: int | None
    api_key_name: str | None
    mode: str  # sync | async
    params: PredictParams
    prepared: PreparedInput
    submitted_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    future: Future = field(default_factory=Future)


@dataclass
class JobResult:
    task_id: str
    status: str  # succeeded | failed
    predictions: list[dict[str, Any]] | None = None
    inference_ms: int = 0
    queue_ms: int = 0
    error: str | None = None


class QueueFullError(RuntimeError):
    pass


class InferenceQueue:
    def __init__(self, model_manager, db_writer, runtime_config):
        self._model_manager = model_manager
        self._writer = db_writer
        self._cfg = runtime_config
        self._q: queue.Queue = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._stop = threading.Event()
        self._active = 0
        self._active_lock = threading.Lock()

        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._wait_ms_hist: deque[int] = deque(maxlen=1000)
        self._infer_ms_hist: deque[int] = deque(maxlen=1000)

    # ------------------------------------------------------------------ lifecycle
    def start(self, n_workers: int, max_queue_size: int) -> None:
        self._q = queue.Queue(maxsize=max_queue_size)
        for i in range(n_workers):
            t = threading.Thread(target=self._worker_loop, name=f"infer-worker-{i}", daemon=True)
            t.start()
            self._workers.append(t)
        logger.info("推理队列已启动: %d workers, max_queue=%d", n_workers, max_queue_size)

    def shutdown(self, timeout: float = 90.0) -> None:
        self._stop.set()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and self._active > 0:
            time.sleep(0.2)
        for t in self._workers:
            t.join(timeout=5.0)

    def running(self) -> bool:
        return not self._stop.is_set()

    # ------------------------------------------------------------------ submit
    def submit(
        self,
        model_key: str,
        params: PredictParams,
        prepared: PreparedInput,
        api_key_id: int | None,
        api_key_name: str | None,
        mode: str,
    ) -> Job:
        task_id = uuid.uuid4().hex
        job = Job(
            task_id=task_id,
            model_key=model_key,
            api_key_id=api_key_id,
            api_key_name=api_key_name,
            mode=mode,
            params=params,
            prepared=prepared,
        )
        try:
            self._q.put_nowait(job)
        except queue.Full as e:
            raise QueueFullError("服务繁忙，等待队列已满，请稍后重试") from e
        self._submitted += 1
        self._writer.upsert_task(
            task_id,
            api_key_id=api_key_id,
            mode=mode,
            status="queued",
            model=model_key,
            symbol=prepared.symbol,
            pred_len=params.pred_len,
            params={
                "temperature": params.temperature,
                "top_k": params.top_k,
                "top_p": params.top_p,
                "sample_count": params.sample_count,
                "lookback": prepared.lookback,
            },
        )
        return job

    # ------------------------------------------------------------------ workers
    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            with self._active_lock:
                self._active += 1
            try:
                jobs = self._maybe_collect_batch(job)
                if len(jobs) > 1:
                    self._run_batch(jobs)
                else:
                    self._run_single(jobs[0])
            except Exception:  # noqa: BLE001 - never kill the worker
                logger.exception("worker 执行异常")
            finally:
                with self._active_lock:
                    self._active -= 1

    def _maybe_collect_batch(self, first: Job) -> list[Job]:
        if not self._cfg.get_bool("enable_dynamic_batch"):
            return [first]
        window_s = self._cfg.get_int("dynamic_batch_window_ms") / 1000.0
        deadline = time.monotonic() + window_s
        group: list[Job] = [first]
        sig = first.prepared.batch_signature(first.params)
        while time.monotonic() < deadline and len(group) < 16:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                other = self._q.get(timeout=min(0.01, remaining) if remaining > 0 else 0.01)
            except queue.Empty:
                break
            if other.prepared.batch_signature(other.params) == sig:
                group.append(other)
            else:
                # unrelated job: put it back at the front is not possible with
                # queue.Queue; execute it after the group by stashing locally.
                group.append(other)
                break
        return group

    def _mark_running(self, job: Job) -> None:
        job.started_at = time.monotonic()
        queue_ms = int((job.started_at - job.submitted_at) * 1000)
        self._wait_ms_hist.append(queue_ms)
        self._writer.upsert_task(job.task_id, status="running", queue_ms=queue_ms, started_at=_utcnow())

    def _finish(self, job: Job, result: JobResult) -> None:
        if result.status == "succeeded":
            self._completed += 1
        else:
            self._failed += 1
        payload = {
            "status": result.status,
            "queue_ms": result.queue_ms,
            "inference_ms": result.inference_ms,
            "finished_at": _utcnow(),
        }
        if result.status == "succeeded" and result.predictions is not None:
            payload["result"] = {"predictions": result.predictions}
        if result.error:
            payload["error"] = result.error[:2000]
        self._writer.upsert_task(job.task_id, **payload)
        try:
            if result.status == "succeeded":
                job.future.set_result(result)
            else:
                job.future.set_exception(RuntimeError(result.error))
        except Exception:  # noqa: BLE001 - future may already be cancelled/timeout
            logger.debug("任务 %s 的 future 已不可设置（可能已被取消）", job.task_id)

    def _run_single(self, job: Job) -> None:
        self._mark_running(job)
        queue_ms = int((job.started_at - job.submitted_at) * 1000)
        try:
            with self._model_manager.acquire(job.model_key) as loaded:
                pred_df, inference_ms = run_prediction(
                    loaded.predictor, job.prepared, job.params
                )
                check_prediction_sane(pred_df, job.params.pred_len)
            self._infer_ms_hist.append(inference_ms)
            self._finish(
                job,
                JobResult(
                    task_id=job.task_id,
                    status="succeeded",
                    predictions=format_predictions(pred_df),
                    inference_ms=inference_ms,
                    queue_ms=queue_ms,
                ),
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("任务 %s 推理失败", job.task_id)
            self._finish(
                job,
                JobResult(
                    task_id=job.task_id,
                    status="failed",
                    error=str(e)[:2000],
                    queue_ms=queue_ms,
                ),
            )

    def _run_batch(self, jobs: list[Job]) -> None:
        """Dynamic batch: first N same-signature jobs run via predict_batch,
        any trailing unrelated job is executed separately."""
        sig = jobs[0].prepared.batch_signature(jobs[0].params)
        group, rest = [jobs[0]], []
        for j in jobs[1:]:
            (group if j.prepared.batch_signature(j.params) == sig else rest).append(j)

        for j in group:
            self._mark_running(j)
        params = group[0].params
        try:
            with self._model_manager.acquire(group[0].model_key) as loaded:
                pred_dfs, inference_ms = run_prediction_batch(
                    loaded.predictor, [j.prepared for j in group], params
                )
            self._infer_ms_hist.append(inference_ms)
            for j, pred_df in zip(group, pred_dfs):
                check_prediction_sane(pred_df, j.params.pred_len)
                self._finish(
                    j,
                    JobResult(
                        task_id=j.task_id,
                        status="succeeded",
                        predictions=format_predictions(pred_df),
                        inference_ms=inference_ms,
                        queue_ms=int((j.started_at - j.submitted_at) * 1000),
                    ),
                )
        except Exception as e:  # noqa: BLE001
            logger.exception("合批推理失败 (%d jobs)", len(group))
            for j in group:
                self._finish(
                    j,
                    JobResult(
                        task_id=j.task_id,
                        status="failed",
                        error=str(e)[:2000],
                        queue_ms=int((j.started_at - j.submitted_at) * 1000),
                    ),
                )
        for j in rest:
            self._run_single(j)

    # ------------------------------------------------------------------ metrics
    def metrics(self) -> dict[str, Any]:
        with self._active_lock:
            active = self._active
        return {
            "queued": self._q.qsize(),
            "active": active,
            "workers": len(self._workers),
            "submitted": self._submitted,
            "completed": self._completed,
            "failed": self._failed,
            "wait_ms_p50": percentile(self._wait_ms_hist, 0.50),
            "wait_ms_p95": percentile(self._wait_ms_hist, 0.95),
            "infer_ms_p50": percentile(self._infer_ms_hist, 0.50),
            "infer_ms_p95": percentile(self._infer_ms_hist, 0.95),
        }


def _utcnow():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)
