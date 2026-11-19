"""Export index daily K-lines from MongoDB (index_daily_quotes) to qlib bin format.

Separate self-contained dump (its own calendar + instruments + benchmark) so the
stock dump at cn_data_as stays untouched:

    /workspace/data/qlib_data/cn_data_indices/
        calendars/day.txt      (000300 trading days, authoritative)
        instruments/all.txt    (all indices incl. sh000300 benchmark)
        features/<inst>/{open,close,high,low,volume,vwap,factor}.day.bin

Indices need NO price adjustment (index series are already continuous across
constituent rebalances). Null open/high/low on the earliest records fall back
to close. vwap falls back to close; factor is constant 1.

Run inside the kronos-finetune container:
    python finetune/data_prep/export_indices_to_qlib.py --out /workspace/data/qlib_data/cn_data_indices
    python finetune/data_prep/export_indices_to_qlib.py --verify --out /workspace/data/qlib_data/cn_data_indices
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from pymongo import MongoClient

DEFAULT_URI = (
    "mongodb://admin:Z_8cc8V6QzyU8EntG3KG137-59I6yHt3@mongodb:27017/"
    "tradingagentscn?authSource=admin"
)
DEFAULT_DB = "tradingagentscn"

FIELDS = ("open", "close", "high", "low", "volume", "vwap", "factor")
BENCH_CODE = "000300"


def prefix_for(symbol: str) -> str:
    if "." in symbol:
        suf = symbol.split(".")[-1].upper()
        if suf in ("SH", "SS"):
            return "sh"
        if suf == "SZ":
            return "sz"
        if suf == "CSI":
            return "csi"
    return "sh"


def write_bin(path: str, start_idx: int, values: np.ndarray):
    arr = np.empty(len(values) + 1, dtype="<f4")
    arr[0] = start_idx
    arr[1:] = values
    arr.tofile(path)


def run_export(args):
    db = MongoClient(args.uri)[args.db]
    coll = db["index_daily_quotes"]
    t0 = time.time()

    codes = sorted(coll.distinct("code"))
    print(f"[index-export] {len(codes)} indices: {codes}")

    # authoritative calendar = benchmark index trade dates
    calendar = sorted(d[:10] for d in coll.distinct("trade_date", {"code": BENCH_CODE}) if d)
    print(f"[index-export] calendar({BENCH_CODE}): {len(calendar)} days {calendar[0]}..{calendar[-1]}")
    cal_idx = {d: i for i, d in enumerate(calendar)}

    os.makedirs(os.path.join(args.out, "features"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "calendars"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "instruments"), exist_ok=True)
    with open(os.path.join(args.out, "calendars", "day.txt"), "w") as f:
        f.write("\n".join(calendar) + "\n")

    inst_lines, stats = [], {"exported": 0, "dropped": 0}
    for code in codes:
        docs = list(coll.find({"code": code}, {"_id": 0}).sort("trade_date", 1))
        if len(docs) < 101:
            stats["dropped"] += 1
            continue
        df = pd.DataFrame(docs)
        df["trade_date"] = df["trade_date"].astype(str).str[:10]
        df = df.drop_duplicates("trade_date", keep="last").set_index("trade_date")
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                df[col] = df["close"]
        # earliest records may carry null open/high/low -> close fallback
        for col in ("open", "high", "low"):
            df[col] = df[col].fillna(df["close"])
        df["volume"] = pd.to_numeric(df.get("volume", 0.0), errors="coerce").fillna(0.0)

        sub_cal = [d for d in calendar if df.index[0] <= d <= df.index[-1]]
        aligned = df.reindex(sub_cal)

        inst = f"{prefix_for(docs[0].get('symbol') or '')}{code}".lower()
        fdir = os.path.join(args.out, "features", inst)
        os.makedirs(fdir, exist_ok=True)
        first = cal_idx[sub_cal[0]]
        for field in FIELDS:
            if field == "factor":
                vals = np.ones(len(aligned), dtype=np.float32)
            elif field == "vwap":
                vals = aligned["close"].values.astype(np.float32)
            else:
                vals = aligned[field].values.astype(np.float32)
            write_bin(os.path.join(fdir, f"{field}.day.bin"), first, vals)

        valid = aligned["close"].notna()
        start_day = sub_cal[int(np.argmax(valid.values))]
        end_day = sub_cal[len(sub_cal) - 1 - int(np.argmax(valid.values[::-1]))]
        inst_lines.append(f"{inst}\t{start_day}\t{end_day}")
        stats["exported"] += 1
        print(f"  {inst}: {len(df)} rows, {df.index[0]}..{df.index[-1]}")

    with open(os.path.join(args.out, "instruments", "all.txt"), "w") as f:
        f.write("\n".join(sorted(inst_lines)) + "\n")

    stats["elapsed_sec"] = round(time.time() - t0, 1)
    stats["calendar_days"] = len(calendar)
    stats["benchmark"] = f"sh{BENCH_CODE}"
    print(f"[index-export] done in {stats['elapsed_sec']}s: {stats}")
    with open(os.path.join(args.out, "export_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def run_verify(args):
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D

    qlib.init(provider_uri=args.out, region=REG_CN)
    cal = D.calendar(freq="day")
    insts = D.list_instruments(D.instruments("all"), as_list=True)
    print(f"[index-verify] calendar days={len(cal)} {cal[0].date()}..{cal[-1].date()}")
    print(f"[index-verify] instruments={len(insts)}: {sorted(insts)}")
    df = D.features(list(insts), ["$close"], freq="day")
    bad = 0
    for inst in insts:
        closes = df.xs(inst, level=0)["$close"].dropna()
        ret = closes.pct_change().dropna().abs()
        n_bad = int((ret > 0.11).sum())  # indices have no price limit; >11% is suspicious
        if n_bad:
            print(f"[index-verify][WARN] {inst}: {n_bad} days |ret|>11%")
            bad += n_bad
    print(f"[index-verify] |ret|>11% outliers total: {bad}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/workspace/data/qlib_data/cn_data_indices")
    parser.add_argument("--uri", default=os.environ.get("MONGO_URI", DEFAULT_URI))
    parser.add_argument("--db", default=os.environ.get("MONGO_DB", DEFAULT_DB))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        run_verify(args)
    else:
        run_export(args)


if __name__ == "__main__":
    sys.exit(main())
