"""Background DB writer: batches hot-path writes off the request path.

All writes (request logs, task upserts, api-key usage counters) go through an
in-memory queue drained by a single writer thread, so inference workers and
request handlers never contend on SQLite write locks. FIFO order is preserved,
which keeps task status transitions ordered.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from sqlalchemy import delete, func, select, update

from .db_models import ApiKey, RequestLog, RuntimeConfigRow, TaskRecord, utcnow

log = logging.getLogger("kronos.db_writer")

_FLUSH_INTERVAL_S = 1.0
_FLUSH_BATCH = 100


class DbWriter:
    """Single background thread performing all DB writes for the service."""

    def __init__(self, session_factory, retention: dict[str, int] | None = None):
        self._session_factory = session_factory
        self._queue: queue.Queue = queue.Queue(maxsize=20000)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._custom_cleanups: list[Callable[[], None]] = []
        # retention config: {"request_log_days": int, "task_hours": int}
        self._retention = retention or {}
        self._last_cleanup = datetime.min.replace(tzinfo=timezone.utc)

    # ------------------------------------------------------------------ API
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="db-writer", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def log_request(self, **fields: Any) -> None:
        """Enqueue a RequestLog row (keyword args match RequestLog columns)."""
        self._put(("request_log", fields))

    def upsert_task(self, task_id: str, **fields: Any) -> None:
        """Enqueue a TaskRecord upsert. FIFO keeps status transitions ordered."""
        self._put(("task_upsert", {"task_id": task_id, **fields}))

    def touch_key(self, api_key_id: int) -> None:
        """Enqueue last_used_at / total_calls bump for an api key."""
        self._put(("touch_key", api_key_id))

    def set_retention(self, request_log_days: int, task_hours: int) -> None:
        self._retention = {"request_log_days": request_log_days, "task_hours": task_hours}

    # -------------------------------------------------------------- internals
    def _put(self, item: tuple) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            log.warning("db writer queue full, dropping %s", item[0])

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                batch = self._drain()
                if batch:
                    self._apply(batch)
                self._maybe_cleanup()
            except Exception:  # noqa: BLE001 - writer must never die
                log.exception("db writer iteration failed")
            self._stop.wait(_FLUSH_INTERVAL_S)
        # final flush on shutdown
        batch = self._drain()
        if batch:
            try:
                self._apply(batch)
            except Exception:
                log.exception("db writer final flush failed")

    def _drain(self) -> list[tuple]:
        batch: list[tuple] = []
        try:
            while len(batch) < _FLUSH_BATCH:
                item = self._queue.get_nowait()
                batch.append(item)
        except queue.Empty:
            pass
        return batch

    def _apply(self, batch: list[tuple]) -> None:
        session = self._session_factory()
        try:
            touched_keys: set[int] = set()
            # task rows staged in this batch are not yet in the identity map,
            # so keep a local handle to append updates onto them
            staged_tasks: dict[str, TaskRecord] = {}
            for kind, payload in batch:
                if kind == "request_log":
                    session.add(RequestLog(**payload))
                elif kind == "task_upsert":
                    self._upsert_task(session, staged_tasks, payload)
                elif kind == "touch_key":
                    touched_keys.add(payload)
            if touched_keys:
                session.execute(
                    update(ApiKey)
                    .where(ApiKey.id.in_(touched_keys))
                    .values(last_used_at=utcnow(), total_calls=ApiKey.total_calls + 1)
                )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _upsert_task(self, session, staged: dict, payload: dict) -> None:
        task_id = payload.pop("task_id")
        row = staged.get(task_id) or session.get(TaskRecord, task_id)
        if row is None:
            row = TaskRecord(task_id=task_id)
            session.add(row)
        staged[task_id] = row
        for k, v in payload.items():
            if k in ("params", "result") and v is not None and not isinstance(v, str):
                v = json.dumps(v, ensure_ascii=False, default=str)
            setattr(row, k, v)

    def _maybe_cleanup(self) -> None:
        if not self._retention:
            return
        now = datetime.now(timezone.utc)
        if (now - self._last_cleanup) < timedelta(hours=1):
            return
        self._last_cleanup = now
        log_days = int(self._retention.get("request_log_days", 30))
        task_hours = int(self._retention.get("task_hours", 24))
        cutoff_logs = now - timedelta(days=log_days)
        cutoff_tasks = now - timedelta(hours=task_hours)
        session = self._session_factory()
        try:
            r1 = session.execute(
                delete(RequestLog).where(RequestLog.ts < cutoff_logs)
            )
            r2 = session.execute(
                delete(TaskRecord).where(
                    TaskRecord.created_at < cutoff_tasks,
                    TaskRecord.status.in_(["succeeded", "failed"]),
                )
            )
            session.commit()
            if (r1.rowcount or 0) + (r2.rowcount or 0) > 0:
                log.info(
                    "retention cleanup removed %d request logs, %d tasks",
                    r1.rowcount or 0,
                    r2.rowcount or 0,
                )
        except Exception:
            session.rollback()
            log.exception("retention cleanup failed")
        finally:
            session.close()
