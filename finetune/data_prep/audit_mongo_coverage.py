"""Audit MongoDB A-share daily K-line coverage before exporting to qlib format.

Answers the questions the export/config steps depend on:
  - how many docs / symbols / date range (period=daily)
  - (data_source x adjust) distribution -> dedup rule impact
  - pre_close / OHLC null rates -> hfq adjustment feasibility
  - per-symbol row counts -> how many symbols survive the 101-day minimum
  - full_symbol suffix style -> exchange prefix mapping (sh/sz/bj)
  - index_daily_quotes schema -> how to build the SH000300 benchmark

Read-only. Run inside the kronos-finetune container:

    python finetune/data_prep/audit_mongo_coverage.py [--out /workspace/data/audit/report.json]

Environment overrides: MONGO_URI, MONGO_DB.
"""

import argparse
import datetime
import json
import os
import sys
from collections import Counter

from pymongo import MongoClient

DEFAULT_URI = (
    "mongodb://admin:Z_8cc8V6QzyU8EntG3KG137-59I6yHt3@mongodb:27017/"
    "tradingagentscn?authSource=admin"
)
DEFAULT_DB = "tradingagentscn"

MIN_ROWS_FOR_TRAIN = 101  # lookback(90) + predict(10) + 1, mirrors qlib_data_preprocess.py


def md(conn_kwargs):
    return MongoClient(**conn_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/workspace/data/audit/mongo_audit_report.json")
    args = parser.parse_args()

    uri = os.environ.get("MONGO_URI", DEFAULT_URI)
    db_name = os.environ.get("MONGO_DB", DEFAULT_DB)

    client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    db = client[db_name]
    client.admin.command("ping")
    print(f"[audit] connected: {db_name}")

    report = {"generated_at": datetime.datetime.utcnow().isoformat()}

    # ------------------------------------------------------------------
    # stock_daily_quotes
    # ------------------------------------------------------------------
    coll = db["stock_daily_quotes"]
    q = {"period": "daily"}

    total = coll.count_documents(q)
    report["daily_total_docs"] = total
    print(f"[audit] stock_daily_quotes period=daily: {total:,} docs")

    # (data_source x adjust) distribution
    src_adj = list(
        coll.aggregate(
            [
                {"$match": q},
                {
                    "$group": {
                        "_id": {"src": "$data_source", "adj": {"$ifNull": ["$adjust", "<missing>"]}},
                        "n": {"$sum": 1},
                    }
                },
                {"$sort": {"n": -1}},
            ],
            allowDiskUse=True,
        )
    )
    report["by_source_adjust"] = [
        {"source": d["_id"]["src"], "adjust": d["_id"]["adj"], "n": d["n"]} for d in src_adj
    ]
    print("[audit] data_source x adjust:")
    for d in report["by_source_adjust"]:
        print(f"    {d['source']:<10} adjust={d['adjust']:<10} {d['n']:>12,}")

    # market field distribution (if present)
    markets = list(coll.aggregate([{"$match": q}, {"$group": {"_id": "$market", "n": {"$sum": 1}}}], allowDiskUse=True))
    report["by_market"] = [{"market": d["_id"], "n": d["n"]} for d in markets]
    print(f"[audit] market field distribution: {report['by_market']}")

    # date range + per-year histogram
    date_range = list(coll.aggregate([{"$match": q}, {"$group": {"_id": None, "mn": {"$min": "$trade_date"}, "mx": {"$max": "$trade_date"}}}], allowDiskUse=True))
    report["date_range"] = date_range[0] if date_range else None
    print(f"[audit] trade_date range: {report['date_range']}")

    by_year = list(
        coll.aggregate(
            [
                {"$match": q},
                {"$project": {"y": {"$substrBytes": ["$trade_date", 0, 4]}}},
                {"$group": {"_id": "$y", "n": {"$sum": 1}, "symbols": {"$addToSet": "$symbol"}}},
                {"$project": {"n": 1, "n_symbols": {"$size": "$symbols"}}},
                {"$sort": {"_id": 1}},
            ],
            allowDiskUse=True,
        )
    )
    # n_symbols via addToSet over 13M docs is memory heavy; fall back to counting via $group separately if this fails.
    report["docs_by_year"] = [{"year": d["_id"], "n": d["n"], "n_symbols": d.get("n_symbols")} for d in by_year]
    print("[audit] docs by year:")
    for d in report["docs_by_year"]:
        print(f"    {d['year']}: {d['n']:>12,} docs, ~{d['n_symbols']} symbols")

    # null rates for adjustment-critical fields
    null_stats = list(
        coll.aggregate(
            [
                {"$match": q},
                {
                    "$group": {
                        "_id": None,
                        "n": {"$sum": 1},
                        "pre_close_null": {"$sum": {"$cond": [{"$in": [{"$ifNull": ["$pre_close", None]}, [None]]}, 1, 0]}},
                        "open_null": {"$sum": {"$cond": [{"$in": [{"$ifNull": ["$open", None]}, [None]]}, 1, 0]}},
                        "close_null": {"$sum": {"$cond": [{"$in": [{"$ifNull": ["$close", None]}, [None]]}, 1, 0]}},
                        "volume_null": {"$sum": {"$cond": [{"$in": [{"$ifNull": ["$volume", None]}, [None]]}, 1, 0]}},
                        "amount_null": {"$sum": {"$cond": [{"$in": [{"$ifNull": ["$amount", None]}, [None]]}, 1, 0]}},
                    }
                },
            ],
            allowDiskUse=True,
        )
    )
    report["null_stats"] = null_stats[0] if null_stats else None
    print(f"[audit] null stats: {report['null_stats']}")

    # per-symbol row counts
    sym_counts = list(
        coll.aggregate(
            [
                {"$match": q},
                {"$group": {"_id": "$symbol", "n": {"$sum": 1}, "mn": {"$min": "$trade_date"}, "mx": {"$max": "$trade_date"}}},
                {"$group": {
                    "_id": None,
                    "symbols": {"$sum": 1},
                    "ge_101": {"$sum": {"$cond": [{"$gte": ["$n", MIN_ROWS_FOR_TRAIN]}, 1, 0]}},
                    "ge_500": {"$sum": {"$cond": [{"$gte": ["$n", 500]}, 1, 0]}},
                    "ge_1000": {"$sum": {"$cond": [{"$gte": ["$n", 1000]}, 1, 0]}},
                }},
            ],
            allowDiskUse=True,
        )
    )
    report["symbol_rowcount_summary"] = sym_counts[0] if sym_counts else None
    print(f"[audit] symbol rowcount summary: {report['symbol_rowcount_summary']}")

    # full_symbol suffix style (sample)
    sample_syms = list(coll.aggregate([{"$match": q}, {"$sample": {"size": 20}}, {"$project": {"_id": 0, "symbol": 1, "full_symbol": 1, "code": 1}}]))
    report["symbol_samples"] = sample_syms
    suffixes = Counter()
    for doc in coll.aggregate([{"$match": q}, {"$sample": {"size": 5000}}, {"$project": {"full_symbol": 1}}]):
        fs = doc.get("full_symbol") or ""
        suffixes[fs[-3:].upper() if "." in fs else "<none>"] += 1
    report["full_symbol_suffix_distribution"] = dict(suffixes)
    print(f"[audit] full_symbol suffix distribution (5k sample): {dict(suffixes)}")

    # sample document (full schema)
    report["sample_doc"] = coll.find_one(q, {"_id": 0})
    print("[audit] sample doc keys:", sorted(report["sample_doc"].keys()) if report["sample_doc"] else None)

    # ------------------------------------------------------------------
    # index_daily_quotes (benchmark)
    # ------------------------------------------------------------------
    try:
        icoll = db["index_daily_quotes"]
        idx_total = icoll.estimated_document_count()
        idx_codes = list(icoll.aggregate([{"$group": {"_id": "$code", "n": {"$sum": 1}}}, {"$sort": {"n": -1}}, {"$limit": 20}]))
        idx_sample = icoll.find_one({}, {"_id": 0})
        idx_300 = list(icoll.find(
            {"$or": [{"code": {"$regex": "000300"}}, {"symbol": {"$regex": "000300"}}]},
            {"_id": 0},
        ).sort("trade_date", 1).limit(1))
        idx_300_last = list(icoll.find(
            {"$or": [{"code": {"$regex": "000300"}}, {"symbol": {"$regex": "000300"}}]},
            {"_id": 0},
        ).sort("trade_date", -1).limit(1))
        report["index_daily_quotes"] = {
            "total": idx_total,
            "top_codes": [{"code": d["_id"], "n": d["n"]} for d in idx_codes],
            "sample_doc": idx_sample,
            "csi300_first": idx_300[0] if idx_300 else None,
            "csi300_last": idx_300_last[0] if idx_300_last else None,
        }
        print(f"[audit] index_daily_quotes: total={idx_total}, top codes={[d['_id'] for d in idx_codes[:8]]}")
    except Exception as e:  # noqa: BLE001
        report["index_daily_quotes"] = {"error": str(e)}
        print(f"[audit] index_daily_quotes audit failed: {e}")

    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"[audit] report saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
