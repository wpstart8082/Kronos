"""Integration tests against the real Kronos model.

Skipped unless KRONOS_INTEGRATION=1 (weights resolved from ./models or HF cache).
Run:  KRONOS_INTEGRATION=1 pytest tests/service/test_integration_real_model.py -m integration
"""
from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("KRONOS_INTEGRATION") != "1",
                       reason="set KRONOS_INTEGRATION=1 to run real-model tests"),
]


def test_real_sync_predict(client, api_key):
    """Real kronos-base (or fallback) end-to-end prediction."""
    import pandas as pd

    df = pd.read_csv("tests/data/regression_input.csv").tail(256).reset_index(drop=True)
    stamps = pd.to_datetime(df["timestamps"])
    candles = [
        {
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "open": float(r["open"]), "high": float(r["high"]),
            "low": float(r["low"]), "close": float(r["close"]),
            "volume": float(r["volume"]), "amount": float(r["amount"]),
        }
        for ts, (_, r) in zip(stamps, df.iterrows())
    ]
    r = client.post(
        "/v1/predict",
        headers={"X-API-Key": api_key},
        json={"model": "kronos-base", "symbol": "REGRESSION", "data": candles,
              "pred_len": 16, "sample_count": 1},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["predictions"]) == 16
    assert all(p["close"] > 0 for p in body["predictions"])
