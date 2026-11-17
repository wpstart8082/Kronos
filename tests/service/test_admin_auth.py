"""Admin auth: login, JWT guard, brute-force lockout, overview content."""
from __future__ import annotations

from .conftest import ADMIN_PASS, ADMIN_USER


def test_health_endpoints_no_auth(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    r = client.get("/readyz")  # no preload in tests -> ready once queue runs
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_admin_requires_token(client):
    assert client.get("/api/admin/overview").status_code == 401
    assert client.get("/api/admin/keys").status_code == 401
    assert client.get("/api/admin/config").status_code == 401


def test_login_wrong_password(client):
    r = client.post("/api/admin/login", json={"username": ADMIN_USER, "password": "wrong"})
    assert r.status_code == 401


def test_login_success_and_overview(client, admin_token):
    r = client.get("/api/admin/overview", headers={"Authorization": f"Bearer {admin_token}"})
    assert r.status_code == 200
    body = r.json()
    for key in ("service", "queue", "models", "gpu", "requests_1h", "requests_24h",
                "hourly_series", "top_keys", "top_symbols", "system", "config"):
        assert key in body


def test_brute_force_lockout(client):
    for _ in range(5):
        client.post("/api/admin/login", json={"username": ADMIN_USER, "password": "nope"})
    # even the correct password is now locked
    r = client.post("/api/admin/login", json={"username": ADMIN_USER, "password": ADMIN_PASS})
    assert r.status_code == 429


def test_config_get_put(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.get("/api/admin/config", headers=headers)
    assert r.status_code == 200
    cfg = {c["key"]: c for c in r.json()["config"]}
    assert cfg["default_model"]["value"] == "kronos-base"

    r = client.put("/api/admin/config", headers=headers, json={"max_pred_len": 240, "enable_dynamic_batch": True})
    assert r.status_code == 200
    assert "max_pred_len" in r.json()["applied"]

    # invalid value rejected
    r = client.put("/api/admin/config", headers=headers, json={"max_pred_len": 99999})
    assert r.status_code == 422
