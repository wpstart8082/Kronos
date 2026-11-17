"""Predict API: auth, validation, sync predict, rate limit, keys lifecycle."""
from __future__ import annotations

from .conftest import make_daily_candles


def _hdr(key):
    return {"X-API-Key": key}


def test_predict_requires_key(client):
    r = client.post("/v1/predict", json={"data": make_daily_candles(), "pred_len": 8})
    assert r.status_code == 401


def test_predict_invalid_key(client):
    r = client.post("/v1/predict", headers=_hdr("kron-doesnotexist"), json={"data": make_daily_candles(), "pred_len": 8})
    assert r.status_code == 401


def test_predict_validation_errors(client, api_key):
    # pred_len out of schema bounds
    r = client.post("/v1/predict", headers=_hdr(api_key), json={"data": make_daily_candles(), "pred_len": 0})
    assert r.status_code == 422
    # unknown model
    r = client.post("/v1/predict", headers=_hdr(api_key),
                    json={"model": "gpt-4", "data": make_daily_candles(), "pred_len": 8})
    assert r.status_code == 422
    # unparseable timestamp
    bad = make_daily_candles(40)
    bad[0]["timestamp"] = "not-a-date"
    r = client.post("/v1/predict", headers=_hdr(api_key), json={"data": bad, "pred_len": 8})
    assert r.status_code == 422
    # future_timestamps length mismatch
    candles = make_daily_candles(40)
    r = client.post("/v1/predict", headers=_hdr(api_key),
                    json={"data": candles, "pred_len": 8, "future_timestamps": ["2026-12-01"]})
    assert r.status_code == 422
    assert "pred_len" in r.json()["detail"]


def test_models_endpoint(client, api_key, fake_model):
    r = client.get("/v1/models", headers=_hdr(api_key))
    assert r.status_code == 200
    keys = [m["key"] for m in r.json()["models"]]
    assert keys == ["kronos-mini", "kronos-small", "kronos-base"]


def test_sync_predict_happy_path(client, api_key, fake_model):
    candles = make_daily_candles(60)
    r = client.post("/v1/predict", headers=_hdr(api_key),
                    json={"symbol": "300308", "data": candles, "pred_len": 10,
                          "temperature": 1.0, "top_p": 0.9, "sample_count": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["symbol"] == "300308"
    assert body["model"] == "kronos-base"
    assert len(body["predictions"]) == 10
    assert body["input"]["points"] == 60
    assert body["input"]["frequency"] == "daily"
    assert body["timings"]["total_ms"] >= body["timings"]["inference_ms"]
    first = body["predictions"][0]
    assert {"timestamp", "open", "high", "low", "close", "volume", "amount"} <= set(first)


def test_rate_limit_per_minute(client, admin_token, fake_model):
    r = client.post("/api/admin/keys", headers={"Authorization": f"Bearer {admin_token}"},
                    json={"name": "limited", "rate_limit_per_min": 2, "daily_quota": 0})
    key = r.json()["plaintext"]
    codes = []
    for _ in range(4):
        codes.append(
            client.post("/v1/predict", headers=_hdr(key),
                        json={"data": make_daily_candles(40), "pred_len": 4}).status_code
        )
    assert codes[:2] == [200, 200]
    assert 429 in codes[2:]


def test_disabled_key_rejected(client, admin_token, api_key, fake_model):
    headers = {"Authorization": f"Bearer {admin_token}"}
    key_id = client.get("/api/admin/keys", headers=headers).json()["keys"][0]["id"]
    r = client.patch(f"/api/admin/keys/{key_id}", headers=headers, json={"enabled": False})
    assert r.status_code == 200
    assert client.post("/v1/predict", headers=_hdr(api_key),
                       json={"data": make_daily_candles(40), "pred_len": 4}).status_code == 403


def test_revoked_key_rejected(client, admin_token, api_key):
    headers = {"Authorization": f"Bearer {admin_token}"}
    key_id = client.get("/api/admin/keys", headers=headers).json()["keys"][0]["id"]
    assert client.delete(f"/api/admin/keys/{key_id}", headers=headers).status_code == 200
    assert client.post("/v1/predict", headers=_hdr(api_key),
                       json={"data": make_daily_candles(40), "pred_len": 4}).status_code == 403
