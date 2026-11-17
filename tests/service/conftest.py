"""Shared fixtures: hermetic app instance with temp dirs and fake predictor."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICE_DIR = REPO_ROOT / "kronos_service"
for p in (str(REPO_ROOT), str(SERVICE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

ADMIN_USER = "admin"
ADMIN_PASS = "test-pass-123"


@pytest.fixture()
def client(tmp_path):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.settings import Settings

    settings = Settings(
        _env_file=None,
        env="test",
        instance_dir=tmp_path / "instance",
        log_dir=tmp_path / "logs",
        models_dir=tmp_path / "models",
        admin_username=ADMIN_USER,
        admin_password=ADMIN_PASS,
        jwt_secret="test-jwt-secret-0123456789abcdef0123456789",
    )
    app = create_app(settings, preload=[])
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def admin_token(client):
    r = client.post("/api/admin/login", json={"username": ADMIN_USER, "password": ADMIN_PASS})
    assert r.status_code == 200, r.text
    return r.json()["token"]


@pytest.fixture()
def api_key(client, admin_token):
    r = client.post(
        "/api/admin/keys",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"name": "pytest-key", "note": "", "rate_limit_per_min": 0, "daily_quota": 0},
    )
    assert r.status_code == 200, r.text
    return r.json()["plaintext"]


@pytest.fixture()
def other_api_key(client, admin_token):
    r = client.post(
        "/api/admin/keys",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"name": "pytest-key-2", "note": "", "rate_limit_per_min": 0, "daily_quota": 0},
    )
    assert r.status_code == 200, r.text
    return r.json()["plaintext"]


class FakePredictor:
    """Deterministic stand-in for KronosPredictor."""

    def predict(self, df, x_timestamp, y_timestamp, pred_len, T=1.0, top_k=0, top_p=0.9,
                sample_count=1, verbose=False):
        last_close = float(df["close"].iloc[-1])
        n = len(y_timestamp)
        step = max(last_close * 0.001, 0.01)
        closes = [last_close + step * (i + 1) for i in range(n)]
        data = {
            "open": [c - step / 2 for c in closes],
            "high": [c + step for c in closes],
            "low": [c - step for c in closes],
            "close": closes,
            "volume": [1000.0] * n,
            "amount": [c * 1000 for c in closes],
        }
        return pd.DataFrame(data, index=pd.DatetimeIndex(pd.to_datetime(list(y_timestamp))))

    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, **kw):
        return [self.predict(df, x, y, pred_len, **kw) for df, x, y in
                zip(df_list, x_timestamp_list, y_timestamp_list)]


@pytest.fixture()
def fake_model(client, monkeypatch):
    """Replace model loading with the fake predictor."""
    from app.model_manager import LoadedModel

    app = client.app
    manager = app.state.model_manager

    def fake_load(key):
        if key not in manager._loaded:
            import time as _t

            manager._loaded[key] = LoadedModel(
                key=key, predictor=FakePredictor(), device="cpu",
                max_context=512, loaded_at=_t.time(), load_ms=1,
            )
        return manager._loaded[key]

    monkeypatch.setattr(manager, "load", fake_load)
    return manager


def make_daily_candles(n: int = 60, start: str = "2026-09-01") -> list[dict]:
    """n daily bars (business days) around price 10 with small oscillation."""
    ts = pd.bdate_range(start=start, periods=n)
    out = []
    price = 10.0
    for t in ts:
        price *= 1.001
        out.append({
            "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
            "open": round(price * 0.999, 4),
            "high": round(price * 1.01, 4),
            "low": round(price * 0.99, 4),
            "close": round(price, 4),
            "volume": 10000.0,
            "amount": round(price * 10000, 2),
        })
    return out
