"""Dual-model asset routing: symbol inference + /v1/predict routing behaviour."""
from __future__ import annotations

import pytest

from app.asset_routing import infer_asset_type
from .conftest import make_daily_candles

INDEX_SYMBOLS = [
    "sh000300", "SH000300", "sh000001", "sh000905", "sh000852", "sh000688",
    "000300.SH", "000905.SH", "000001.SH",
    "sz399001", "SZ399006", "399001.SZ", "399016.SZ",
    "399001", "399006",  # bare 399xxx: no SZ stocks in this range
    "csi000922", "CSIH00922", "000922.CSI",
    "999999", "sh999999",  # 上证指数行情别名
]
STOCK_SYMBOLS = [
    "600519", "601318", "688111", "300308", "002594", "003816",
    "600519.SH", "688981.SH", "000001.SZ", "300750.SZ",
    "sh600519", "SH688111", "sz000001", "SZ300750",  # prefixed stock codes
]
UNKNOWN_SYMBOLS = [
    None, "", "000300", "000922",  # bare 000xxx ambiguous (SZ stocks share the range)
    "510300", "159915",  # ETF/基金 -> not index, not stock
    "ETHUSDT", "AAPL",
]


@pytest.mark.parametrize("symbol", INDEX_SYMBOLS)
def test_infer_index(symbol):
    assert infer_asset_type(symbol) == "index"


@pytest.mark.parametrize("symbol", STOCK_SYMBOLS)
def test_infer_stock(symbol):
    assert infer_asset_type(symbol) == "stock"


@pytest.mark.parametrize("symbol", UNKNOWN_SYMBOLS)
def test_infer_unknown(symbol):
    assert infer_asset_type(symbol) is None


def _hdr(key):
    return {"X-API-Key": key}


def _post(client, key, payload):
    return client.post("/v1/predict", headers=_hdr(key), json=payload)


def test_route_index_symbol_to_zeroshot(client, api_key, fake_model):
    r = _post(client, api_key, {"symbol": "sh000300", "data": make_daily_candles(), "pred_len": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "kronos-base"
    assert body["asset_type"] == "index"


def test_route_stock_symbol_to_finetuned(client, api_key, fake_model):
    r = _post(client, api_key, {"symbol": "600519", "data": make_daily_candles(), "pred_len": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "kronos-as-base"
    assert body["asset_type"] == "stock"


def test_route_bare_000_falls_back_to_default(client, api_key, fake_model):
    # 裸 000xxx 无法区分深市股票与指数 -> 默认模型, asset_type 回显 null
    r = _post(client, api_key, {"symbol": "000300", "data": make_daily_candles(), "pred_len": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "kronos-base"
    assert body["asset_type"] is None


def test_asset_type_overrides_symbol(client, api_key, fake_model):
    # 显式 asset_type 优先于 symbol 识别: 000300 + asset_type=index -> 指数模型
    r = _post(client, api_key, {"symbol": "000300", "asset_type": "index",
                                "data": make_daily_candles(), "pred_len": 5})
    assert r.status_code == 200, r.text
    assert r.json()["model"] == "kronos-base"


def test_explicit_model_skips_routing(client, api_key, fake_model):
    r = _post(client, api_key, {"symbol": "sh000300", "model": "kronos-base",
                                "data": make_daily_candles(), "pred_len": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "kronos-base"
    assert body["asset_type"] is None


def test_routing_can_be_disabled_via_config(client, admin_token, api_key, fake_model):
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.put("/api/admin/config", headers=headers, json={"enable_asset_routing": False})
    assert r.status_code == 200, r.text
    r = _post(client, api_key, {"symbol": "sh000300", "data": make_daily_candles(), "pred_len": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "kronos-base"  # default_model fallback
    assert body["asset_type"] is None


def test_undeployed_routed_model_returns_503(client, api_key, monkeypatch):
    """路由到权重缺失且无 HF 回退的模型 -> 503 且信息清晰(而非推理期报错)。"""
    from app.model_manager import ModelManager

    def fake_availability(self, key):
        return {"location": "hub"}  # no local weights anywhere

    monkeypatch.setattr(ModelManager, "availability", fake_availability)
    # kronos-as-base has empty hf ids -> cannot download -> 503
    r = _post(client, api_key, {"symbol": "600519", "data": make_daily_candles(), "pred_len": 5})
    assert r.status_code == 503
    assert "权重未部署" in str(r.json()["detail"])
