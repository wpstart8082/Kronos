"""Async task lifecycle and isolation between keys."""
from __future__ import annotations

import time

from .conftest import make_daily_candles


def _poll_task(client, key, task_id, timeout=15.0):
    """Poll a task; tolerates 404 while the db-writer batch flushes (~1s)."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        r = client.get(f"/v1/tasks/{task_id}", headers={"X-API-Key": key})
        if r.status_code == 200:
            last = r.json()
            if last["status"] in ("succeeded", "failed"):
                return last
        time.sleep(0.15)
    return last


def test_async_task_lifecycle(client, api_key, fake_model):
    r = client.post("/v1/predict/async",
                    headers={"X-API-Key": api_key},
                    json={"symbol": "000300", "data": make_daily_candles(60), "pred_len": 8})
    assert r.status_code == 202
    task_id = r.json()["task_id"]

    body = _poll_task(client, api_key, task_id)
    assert body is not None and body["status"] == "succeeded", body
    assert len(body["result"]["predictions"]) == 8
    assert body["inference_ms"] >= 0


def test_task_requires_auth_and_isolation(client, api_key, other_api_key, fake_model):
    assert client.get("/v1/tasks/whatever").status_code == 401
    r = client.post("/v1/predict/async", headers={"X-API-Key": api_key},
                    json={"data": make_daily_candles(40), "pred_len": 4})
    task_id = r.json()["task_id"]
    # another key cannot see this task
    assert client.get(f"/v1/tasks/{task_id}", headers={"X-API-Key": other_api_key}).status_code == 404
    # owner can
    body = _poll_task(client, api_key, task_id)
    assert body is not None


def test_sync_timeout_returns_task_id(client, api_key, fake_model):
    """Sync timeout path: 408 with task_id, result retrievable afterwards."""
    import threading

    fake_model.load("kronos-base")  # populate _loaded with the fake predictor
    original = fake_model._loaded["kronos-base"].predictor
    release = threading.Event()

    class SlowPredictor:
        def predict(self, *a, **kw):
            release.wait(timeout=30)
            return original.predict(*a, **kw)

    fake_model._loaded["kronos-base"].predictor = SlowPredictor()

    # lower sync timeout for the test via runtime config
    client.app.state.cfg._cache["sync_timeout_seconds"] = 1

    result = {}

    def call():
        result["resp"] = client.post("/v1/predict", headers={"X-API-Key": api_key},
                                     json={"data": make_daily_candles(40), "pred_len": 4})

    t = threading.Thread(target=call)
    t.start()
    t.join(timeout=10)
    release.set()

    resp = result["resp"]
    assert resp.status_code == 408
    task_id = resp.json()["detail"]["task_id"]

    body = _poll_task(client, api_key, task_id)
    assert body is not None and body["status"] == "succeeded"
