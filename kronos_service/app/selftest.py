"""Admin self-test suite: smoke prediction, GPU, DB, model directory checks."""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .settings import repo_root

logger = logging.getLogger("kronos.selftest")

SMOKE_DATA = repo_root() / "tests" / "data" / "regression_input.csv"
SMOKE_PRED_LEN = 16


def _ok(name: str, t0: float, detail: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "passed": True,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "detail": detail,
    }


def _fail(name: str, t0: float, error: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": False,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "detail": error,
    }


def check_gpu() -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        import torch

        if not torch.cuda.is_available():
            return _fail(t0=t0, name="GPU 检测", error="CUDA 不可用（torch.cuda.is_available()=False），仅回退 CPU 运行")
        props = torch.cuda.get_device_properties(0)
        detail = (
            f"{torch.cuda.get_device_name(0)} | 显存 {props.total_memory / 1024**3:.1f}GB | "
            f"已用 {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB | CUDA {torch.version.cuda}"
        )
        return _ok("GPU 检测", t0, detail)
    except Exception as e:  # noqa: BLE001
        return _fail("GPU 检测", t0, str(e))


def check_db(session_factory) -> dict[str, Any]:
    from .db_models import RuntimeConfigRow, utcnow

    t0 = time.perf_counter()
    probe_key = "__selftest_probe__"
    try:
        session = session_factory()
        try:
            session.get(RuntimeConfigRow, probe_key)
            session.add(
                RuntimeConfigRow(key=probe_key, value=json.dumps({"ts": utcnow().isoformat()}), updated_at=utcnow())
            )
            session.commit()
            row = session.get(RuntimeConfigRow, probe_key)
            assert row is not None and json.loads(row.value)
            session.delete(row)
            session.commit()
        finally:
            session.close()
        return _ok("数据库读写", t0, "SQLite 写入/读取/删除探测通过")
    except Exception as e:  # noqa: BLE001
        return _fail("数据库读写", t0, str(e))


def check_models_dir(models_dir: Path) -> dict[str, Any]:
    from .model_manager import REGISTRY

    t0 = time.perf_counter()
    try:
        lines, missing = [], []
        for key, meta in REGISTRY.items():
            for label, dirname in (("模型", meta["model_dirname"]), ("分词器", meta["tokenizer_dirname"])):
                path = models_dir / dirname
                if (path / "config.json").is_file() and any(path.glob("*.safetensors")):
                    size_mb = sum(f.stat().st_size for f in path.iterdir() if f.is_file()) / 1024**2
                    lines.append(f"{dirname}({size_mb:.0f}MB)")
                else:
                    lines.append(f"{dirname}(缺失,走HF Hub)")
                    if label == "模型":
                        missing.append(key)
        detail = "; ".join(lines)
        if missing:
            detail += f" | 提示: {', '.join(missing)} 未在本地目录，首次加载需联网"
        return _ok("模型目录扫描", t0, detail)
    except Exception as e:  # noqa: BLE001
        return _fail("模型目录扫描", t0, str(e))


def check_smoke_predict(model_manager, model_key: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        if not SMOKE_DATA.is_file():
            return _fail("模型冒烟预测", t0, f"冒烟数据缺失: {SMOKE_DATA}")
        df = pd.read_csv(SMOKE_DATA)
        x_df = df[["open", "high", "low", "close", "volume", "amount"]].tail(256).reset_index(drop=True)
        x_ts = pd.to_datetime(df["timestamps"]).tail(256).reset_index(drop=True)
        # reuse history tail as future grid (smoke test only checks the pipeline)
        y_ts = pd.to_datetime(df["timestamps"]).tail(SMOKE_PRED_LEN).reset_index(drop=True)
        with model_manager.acquire(model_key) as loaded:
            pred_df, infer_ms = _predict(loaded.predictor, x_df, x_ts, y_ts)
        if pred_df.isna().any().any() or len(pred_df) != SMOKE_PRED_LEN:
            raise RuntimeError("输出包含 NaN 或长度不符")
        last_close = float(pred_df["close"].iloc[-1])
        detail = (
            f"模型 {model_key} 预测 {SMOKE_PRED_LEN} 步成功 | 推理 {infer_ms}ms | "
            f"末根收盘价 {last_close:.4f} | 输入 256 根"
        )
        return _ok("模型冒烟预测", t0, detail)
    except Exception as e:  # noqa: BLE001
        return _fail("模型冒烟预测", t0, str(e))


def _predict(predictor, x_df, x_ts, y_ts):
    t0 = time.perf_counter()
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_ts,
        y_timestamp=y_ts,
        pred_len=SMOKE_PRED_LEN,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=False,
    )
    return pred_df, int((time.perf_counter() - t0) * 1000)


def check_queue(state) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        m = state.queue.metrics()
        if not state.queue.running():
            raise RuntimeError("推理队列未运行")
        return _ok(
            "推理队列",
            t0,
            f"workers={m['workers']} 队列中={m['queued']} 执行中={m['active']} 累计完成={m['completed']}",
        )
    except Exception as e:  # noqa: BLE001
        return _fail("推理队列", t0, str(e))


def check_service_dirs(state) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        issues = []
        for label, path in (("实例目录", state.settings.instance_dir), ("日志目录", state.settings.log_dir)):
            if not path.is_dir():
                issues.append(f"{label}不存在: {path}")
            elif not any(path.iterdir()):
                issues.append(f"{label}为空(首次运行正常)")
        return _ok("服务目录", t0, "; ".join(issues) or "实例/日志目录均可访问")
    except Exception as e:  # noqa: BLE001
        return _fail("服务目录", t0, str(e))


def run_all(state, smoke_model: str) -> list[dict[str, Any]]:
    """Run all checks concurrently; smoke prediction uses its own model load."""
    checks: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(check_gpu): None,
            pool.submit(check_db, state.session_factory): None,
            pool.submit(check_models_dir, state.settings.models_dir): None,
            pool.submit(check_queue, state): None,
            pool.submit(check_service_dirs, state): None,
            pool.submit(check_smoke_predict, state.model_manager, smoke_model): None,
        }
        for fut in futures:
            try:
                checks.append(fut.result(timeout=600))
            except Exception as e:  # noqa: BLE001
                checks.append({"name": "未知检查项", "passed": False, "latency_ms": 0, "detail": str(e)})
    # stable display order
    order = ["GPU 检测", "模型目录扫描", "模型冒烟预测", "推理队列", "数据库读写", "服务目录"]
    checks.sort(key=lambda c: order.index(c["name"]) if c["name"] in order else 99)
    passed = sum(1 for c in checks if c["passed"])
    logger.info("自检完成: %d/%d 通过", passed, len(checks))
    return checks
