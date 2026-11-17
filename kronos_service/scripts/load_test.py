#!/usr/bin/env python3
"""Concurrency smoke/load test for the Kronos prediction service.

Usage:
    python kronos_service/scripts/load_test.py --url http://localhost:8000 --key kron-xxx \
        --concurrency 20 --requests 50 [--pred-len 48]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_candles(n: int = 400) -> list[dict]:
    import pandas as pd

    path = REPO_ROOT / "tests" / "data" / "regression_input.csv"
    df = pd.read_csv(path).tail(n).reset_index(drop=True)
    stamps = pd.to_datetime(df["timestamps"])
    return [
        {
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "open": float(r["open"]), "high": float(r["high"]),
            "low": float(r["low"]), "close": float(r["close"]),
            "volume": float(r["volume"]), "amount": float(r["amount"]),
        }
        for ts, (_, r) in zip(stamps, df.iterrows())
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--key", required=True)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--pred-len", type=int, default=48)
    args = parser.parse_args()

    candles = load_candles()
    payload = {"symbol": "LOADTEST", "data": candles, "pred_len": args.pred_len, "sample_count": 1}
    print(f"目标: {args.url}  并发: {args.concurrency}  总请求: {args.requests}  "
          f"pred_len: {args.pred_len}  输入K线: {len(candles)} 根\n")

    latencies: list[float] = []
    statuses: dict[int, int] = {}
    queue_ms: list[int] = []
    infer_ms: list[int] = []
    lock = __import__("threading").Lock()

    def one(i: int) -> None:
        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=300) as c:
                r = c.post(f"{args.url}/v1/predict",
                           headers={"X-API-Key": args.key}, json=payload)
            with lock:
                statuses[r.status_code] = statuses.get(r.status_code, 0) + 1
                if r.status_code == 200:
                    t = r.json()["timings"]
                    queue_ms.append(t["queue_ms"])
                    infer_ms.append(t["inference_ms"])
        except Exception as e:  # noqa: BLE001
            with lock:
                statuses[0] = statuses.get(0, 0) + 1
                print(f"[{i}] error: {e}")
        finally:
            with lock:
                latencies.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(one, range(args.requests)))
    wall = time.perf_counter() - t0

    lat = sorted(latencies)

    def pct(p):
        return lat[min(int(len(lat) * p), len(lat) - 1)] if lat else 0

    ok = statuses.get(200, 0)
    print(f"完成: {args.requests} 请求 / {wall:.1f}s  吞吐: {args.requests / wall:.2f} req/s")
    print(f"状态码分布: {dict(sorted(statuses.items()))}")
    if ok:
        print(f"端到端延迟(ms):  P50={pct(0.50):.0f}  P95={pct(0.95):.0f}  P99={pct(0.99):.0f}  max={lat[-1]:.0f}")
        print(f"服务端排队(ms):  P50={statistics.median(queue_ms):.0f}  max={max(queue_ms) if queue_ms else 0}")
        print(f"服务端推理(ms):  P50={statistics.median(infer_ms):.0f}  max={max(infer_ms) if infer_ms else 0}")
    return 0 if ok == args.requests else 1


if __name__ == "__main__":
    sys.exit(main())
