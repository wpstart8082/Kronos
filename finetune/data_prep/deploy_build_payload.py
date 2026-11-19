"""One-off: build a /v1/predict payload for 600519 from MongoDB (hfq-adjusted).

Runs INSIDE the kronos-finetune container (has pymongo + mongo network access).
Writes /workspace/deploy_test_payload.json (repo root, bind-mounted to host).
"""

import json

from pymongo import MongoClient

c = MongoClient(
    "mongodb://admin:Z_8cc8V6QzyU8EntG3KG137-59I6yHt3@mongodb:27017/"
    "tradingagentscn?authSource=admin"
)
docs = list(c.tradingagentscn.stock_daily_quotes.find(
    {"symbol": "600519", "period": "daily", "adjust": "none"},
    {"_id": 0, "trade_date": 1, "open": 1, "high": 1, "low": 1, "close": 1,
     "pre_close": 1, "volume": 1, "amount": 1, "data_source": 1}
).sort("trade_date", 1))
print(f"fetched {len(docs)} days, {docs[0]['trade_date']} .. {docs[-1]['trade_date']}")

# hfq adjustment, identical to export_mongo_to_qlib.py
prev = None
factor = 1.0
rows = []
for d in docs:
    ratio = 1.0
    if prev and d.get("pre_close") and prev > 0:
        ratio = prev / d["pre_close"]
    factor *= ratio
    rows.append({
        "timestamp": d["trade_date"],
        "open": d["open"] * factor,
        "high": d["high"] * factor,
        "low": d["low"] * factor,
        "close": d["close"] * factor,
        "volume": float(d["volume"]),
        "amount": float(d["amount"]),
    })
    prev = d["close"]

last90 = rows[-90:]
payload = {
    "symbol": "600519.SH",
    "data": last90,
    "pred_len": 10,
    "temperature": 0.6,
    "top_p": 0.9,
    "sample_count": 3,
}
with open("/workspace/deploy_test_payload.json", "w") as f:
    json.dump(payload, f)
print(f"payload: 90 days, last adjusted close = {last90[-1]['close']:.2f}")
