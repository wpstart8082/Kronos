"""DB-backed runtime configuration with hot-apply semantics.

Initial defaults are defined in code; the admin console can override them.
Keys marked ``restart=True`` only take effect after a service restart
(worker topology cannot be resized at runtime).
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Any

from .db_models import RuntimeConfigRow, utcnow

log = logging.getLogger("kronos.runtime_config")

# key -> (default, type, restart_required, description)
SCHEMA: dict[str, dict[str, Any]] = {
    "default_model": {
        "default": "kronos-base",
        "type": "str",
        "restart": False,
        "desc": "未指定 model 参数时使用的默认模型",
    },
    "preload_models": {
        "default": ["kronos-base"],
        "type": "list",
        "restart": True,
        "desc": "启动时预加载的模型列表（逗号分隔）",
    },
    "max_concurrent_inferences": {
        "default": 1,
        "type": "int",
        "restart": True,
        "desc": "推理并发数（GPU 显存充裕时可调大）",
    },
    "max_queue_size": {
        "default": 200,
        "type": "int",
        "restart": True,
        "desc": "等待队列上限，超出返回 503",
    },
    "sync_timeout_seconds": {
        "default": 120,
        "type": "int",
        "restart": False,
        "desc": "同步接口超时秒数，超时返回 408 + task_id",
    },
    "enable_dynamic_batch": {
        "default": False,
        "type": "bool",
        "restart": False,
        "desc": "同参数请求自动合批（lookback/pred_len/采样参数全相同才合并）",
    },
    "dynamic_batch_window_ms": {
        "default": 50,
        "type": "int",
        "restart": False,
        "desc": "合批等待窗口毫秒数",
    },
    "max_pred_len": {
        "default": 480,
        "type": "int",
        "restart": False,
        "desc": "单次预测最大步数（硬上限 480）",
    },
    "max_sample_count": {
        "default": 8,
        "type": "int",
        "restart": False,
        "desc": "采样路径数上限（硬上限 8）",
    },
    "max_lookback": {
        "default": 512,
        "type": "int",
        "restart": False,
        "desc": "lookback 上限（同时受模型上下文长度约束）",
    },
    "min_lookback": {
        "default": 32,
        "type": "int",
        "restart": False,
        "desc": "lookback 最小值",
    },
    "default_lookback": {
        "default": 400,
        "type": "int",
        "restart": False,
        "desc": "未指定 lookback 时使用的历史窗口长度",
    },
    "max_data_rows": {
        "default": 6000,
        "type": "int",
        "restart": False,
        "desc": "单请求数据行数上限",
    },
    "log_retention_days": {
        "default": 30,
        "type": "int",
        "restart": False,
        "desc": "请求日志保留天数",
    },
    "task_retention_hours": {
        "default": 24,
        "type": "int",
        "restart": False,
        "desc": "已完成任务保留小时数",
    },
}


class RuntimeConfig:
    def __init__(self, session_factory):
        self._session_factory = session_factory
        self._lock = threading.Lock()
        self._cache: dict[str, Any] = {}

    def load(self, overrides: dict[str, Any] | None = None) -> None:
        """Load from DB, seeding defaults; ``overrides`` (from env settings)
        provide initial values for restart-required keys on first boot."""
        with self._lock:
            for key, meta in SCHEMA.items():
                self._cache[key] = meta["default"]
            if overrides:
                for k, v in overrides.items():
                    if k in self._cache and v is not None:
                        self._cache[k] = v
            session = self._session_factory()
            try:
                for row in session.query(RuntimeConfigRow).all():
                    if row.key in SCHEMA:
                        self._cache[row.key] = json.loads(row.value)
                session.commit()
            finally:
                session.close()

    def get(self, key: str) -> Any:
        with self._lock:
            return self._cache[key]

    def get_int(self, key: str) -> int:
        return int(self.get(key))

    def get_bool(self, key: str) -> bool:
        return bool(self.get(key))

    def set_many(self, values: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Validate and persist; returns (applied_keys, restart_required_keys)."""
        applied, restart_keys, errors = [], [], {}
        with self._lock:
            for key, value in values.items():
                if key not in SCHEMA:
                    errors[key] = "未知配置项"
                    continue
                meta = SCHEMA[key]
                try:
                    coerced = _coerce(value, meta["type"])
                    _validate_range(key, coerced)
                except (TypeError, ValueError) as e:
                    errors[key] = f"类型或取值错误: {e}"
                    continue
                self._cache[key] = coerced
                applied.append(key)
                if meta["restart"]:
                    restart_keys.append(key)
            session = self._session_factory()
            try:
                for key in applied:
                    row = session.get(RuntimeConfigRow, key)
                    encoded = json.dumps(self._cache[key], ensure_ascii=False)
                    if row is None:
                        session.add(
                            RuntimeConfigRow(key=key, value=encoded, updated_at=utcnow())
                        )
                    else:
                        row.value = encoded
                        row.updated_at = utcnow()
                session.commit()
            finally:
                session.close()
        if errors:
            raise ValueError(json.dumps(errors, ensure_ascii=False))
        return applied, restart_keys

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "key": key,
                    "value": self._cache[key],
                    "type": meta["type"],
                    "restart_required": meta["restart"],
                    "desc": meta["desc"],
                }
                for key, meta in SCHEMA.items()
            ]


def _coerce(value: Any, type_name: str) -> Any:
    if type_name == "int":
        return int(value)
    if type_name == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if type_name == "list":
        if isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()]
        return [str(v) for v in value]
    return str(value)


def _validate_range(key: str, value: Any) -> None:
    hard_caps = {
        "max_pred_len": (1, 480),
        "max_sample_count": (1, 8),
        "max_lookback": (16, 4096),
        "min_lookback": (8, 4096),
        "default_lookback": (16, 4096),
        "max_data_rows": (100, 200000),
        "max_queue_size": (1, 5000),
        "max_concurrent_inferences": (1, 16),
        "sync_timeout_seconds": (1, 3600),
        "log_retention_days": (1, 365),
        "task_retention_hours": (1, 720),
        "dynamic_batch_window_ms": (0, 1000),
    }
    if key in hard_caps:
        lo, hi = hard_caps[key]
        if not (lo <= value <= hi):
            raise ValueError(f"取值需在 [{lo}, {hi}] 区间内")
