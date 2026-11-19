"""Export A-share daily K-lines from MongoDB to Qlib binary format (cn_data layout).

Pipeline per symbol:
  1. fetch docs (period=daily) sorted by trade_date
  2. dedup per day: prefer adjust=="none", then source priority tushare > akshare > baostock
     (mirrors TradingAgents-CN unified_stock_service.dedup_daily_docs_cn)
  3. keep only days whose chosen doc is unadjusted (adjust=="none"); qfq-legacy-only
     days become gaps (NaN) so they can never splice a price seam into the series
  4. hfq adjustment: ratio_t = pre_close_t / close_{t-1} (missing -> 1), F = cumprod,
     adj_OHLC = raw * F  (true daily returns are preserved: adj_close_t/adj_close_{t-1}
     equals close_t / pre_close_t, including across suspension gaps)
  5. reindex onto the authoritative trading calendar, suspension days = NaN
  6. write features/<inst>/{open,close,high,low,volume,vwap,factor}.bin
     (little-endian float32, first element = calendar start index)

Also exports the SH000300 benchmark from index_daily_quotes and writes
calendars/day.txt + instruments/all.txt.

Run inside the kronos-finetune container:
    python finetune/data_prep/export_mongo_to_qlib.py --out /workspace/data/qlib_data/cn_data_as
    python finetune/data_prep/export_mongo_to_qlib.py --verify --out /workspace/data/qlib_data/cn_data_as
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from pymongo import MongoClient

DEFAULT_URI = (
    "mongodb://admin:Z_8cc8V6QzyU8EntG3KG137-59I6yHt3@mongodb:27017/"
    "tradingagentscn?authSource=admin"
)
DEFAULT_DB = "tradingagentscn"

FIELDS = ("open", "close", "high", "low", "volume", "vwap", "factor")
MIN_ROWS = 101          # lookback(90)+predict(10)+1
PRE_CLOSE_MISSING_LIMIT = 0.05
SRC_PRIORITY = {"tushare": 0, "akshare": 1, "baostock": 2}
CALENDAR_FROM = "2014-01-01"  # unadjusted stock slice starts 2015-01-05


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def dedup_key(doc):
    """Sort key: unadjusted first, then by source priority."""
    adj = doc.get("adjust") or ""
    return (0 if adj == "none" else 1, SRC_PRIORITY.get(doc.get("data_source"), 9))


def symbol_prefix(symbol: str, full_symbol: str) -> str:
    suffix = ""
    if full_symbol and "." in full_symbol:
        suffix = full_symbol.split(".")[-1].upper()
    if suffix in ("SH", "SS"):
        return "sh"
    if suffix == "SZ":
        return "sz"
    if suffix == "BJ":
        return "bj"
    # fallback by first digit
    return {"6": "sh", "9": "sh", "0": "sz", "3": "sz", "4": "bj", "8": "bj"}.get(symbol[:1], "sh")


def write_bin(path: str, start_idx: int, values: np.ndarray):
    arr = np.empty(len(values) + 1, dtype="<f4")
    arr[0] = start_idx
    arr[1:] = values
    arr.tofile(path)


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def build_symbol_frame(coll, symbol: str) -> pd.DataFrame | None:
    """Fetch + dedup + adjust one symbol. Returns DataFrame indexed by date str
    with adjusted OHLC / raw volume, or None if unusable."""
    docs = list(
        coll.find(
            {"symbol": symbol, "period": "daily"},
            {"_id": 0, "trade_date": 1, "open": 1, "high": 1, "low": 1, "close": 1,
             "pre_close": 1, "volume": 1, "amount": 1, "data_source": 1, "adjust": 1},
        ).sort("trade_date", 1)
    )
    if not docs:
        return None

    by_day = defaultdict(list)
    for d in docs:
        by_day[d["trade_date"]].append(d)

    rows, n_legacy_only, n_pre_close_missing = [], 0, 0
    for day in sorted(by_day):
        best = min(by_day[day], key=dedup_key)
        if (best.get("adjust") or "") != "none":
            n_legacy_only += 1
            continue  # legacy qfq day -> gap (NaN), never splice with unadjusted rows
        if best.get("close") in (None, 0) or best.get("open") in (None, 0):
            continue
        if best.get("pre_close") in (None, 0):
            n_pre_close_missing += 1
        rows.append(best)

    if len(rows) < MIN_ROWS:
        return None
    if n_pre_close_missing / len(rows) > PRE_CLOSE_MISSING_LIMIT:
        return None

    df = pd.DataFrame(rows).set_index("trade_date")
    # re-dedupe index (paranoia; unique index should prevent this)
    df = df[~df.index.duplicated(keep="last")]

    prev_close = df["close"].shift(1)
    # hfq factor increment. On ex-div days pre_close_t < close_{t-1} (dividend /
    # split), so prev_close/pre_close > 1 grows the cumulative factor and makes
    # adjusted returns equal true returns: close_t/pre_close_t. (Note the ratio
    # is prev_close/pre_close, NOT its inverse - verified against 002790/300455
    # ex-div days where the true return is close_t/pre_close_t - 1.)
    ratio = prev_close / df["pre_close"]
    n_ratio_forced = int(ratio.isna().sum() + ((ratio <= 0) | ~np.isfinite(ratio)).sum())
    ratio = ratio.where(ratio.notna() & (ratio > 0) & np.isfinite(ratio), 1.0)
    factor = ratio.cumprod()

    out = pd.DataFrame(index=df.index)
    for col in ("open", "high", "low", "close"):
        out[col] = df[col] * factor
    out["volume"] = df["volume"].astype(float)
    vol = df["volume"].where(df["volume"] > 0)
    vwap_raw = (df["amount"] / vol).where(vol.notna())
    out["vwap"] = (vwap_raw * factor).fillna(out["close"])
    out.attrs["n_legacy_only"] = n_legacy_only
    out.attrs["n_ratio_forced"] = n_ratio_forced
    return out


def load_calendar(db) -> list[str]:
    """Authoritative calendar: CSI300 index trade dates; fallback = union of stock dates.
    Trimmed to CALENDAR_FROM (the unadjusted stock slice starts 2015-01-05; a small
    earlier margin keeps the preprocess lookback buffer happy)."""
    try:
        icoll = db["index_daily_quotes"]
        cond = [{"code": {"$regex": "000300"}}, {"symbol": {"$regex": "000300"}}]
        dates = sorted(
            icoll.distinct("trade_date", {"$or": cond})
        )
        dates = [d[:10] for d in dates if d and d >= CALENDAR_FROM]
        if len(dates) > 200:
            print(f"[export] calendar from index_daily_quotes(000300): {len(dates)} days "
                  f"({dates[0]} .. {dates[-1]})")
            return dates
    except Exception as e:  # noqa: BLE001
        print(f"[export] index calendar failed: {e}")
    dates = sorted(db["stock_daily_quotes"].distinct("trade_date", {"period": "daily"}))
    dates = [d[:10] for d in dates if d and d >= CALENDAR_FROM]
    print(f"[export] calendar from stock dates union: {len(dates)} days")
    return dates


def export_index(db, out_dir: str, calendar: list[str]) -> tuple[str, str, str] | None:
    """Export SH000300 benchmark. Returns (inst, start, end) or None."""
    icoll = db["index_daily_quotes"]
    cond = {"$or": [{"code": {"$regex": "000300"}}, {"symbol": {"$regex": "000300"}}]}
    docs = list(icoll.find(cond, {"_id": 0}).sort("trade_date", 1))
    if not docs:
        print("[export][WARN] no 000300 index data found -> benchmark missing")
        return None

    df = pd.DataFrame(docs)
    date_col = "trade_date" if "trade_date" in df.columns else "date"
    df[date_col] = df[date_col].astype(str).str[:10]
    df = df.drop_duplicates(date_col, keep="last").set_index(date_col)
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            df[col] = df["close"]
    # early index records carry null open/high/low -> fall back to close
    for col in ("open", "high", "low"):
        df[col] = df[col].fillna(df["close"])
    df["volume"] = df["volume"] if "volume" in df.columns else 0.0

    df = df[~df.index.duplicated()]
    df = df.reindex(calendar).ffill()  # index should trade every calendar day

    inst = "sh000300"
    fdir = os.path.join(out_dir, "features", inst)
    os.makedirs(fdir, exist_ok=True)
    valid = df["close"].notna()
    first = int(np.argmax(valid.values))
    last = len(df) - 1 - int(np.argmax(valid.values[::-1]))
    for field in FIELDS:
        if field == "factor":
            vals = np.ones(len(df), dtype=np.float32)
        elif field == "vwap":
            vals = df["close"].values.astype(np.float32)
        else:
            vals = df[field].values.astype(np.float32)
        write_bin(os.path.join(fdir, f"{field}.day.bin"), first, vals)
    return inst, calendar[first], calendar[last]


def run_export(args):
    client = MongoClient(args.uri, serverSelectionTimeoutMS=10000)
    db = client[args.db]
    coll = db["stock_daily_quotes"]

    t0 = time.time()
    symbols = sorted(coll.distinct("symbol", {"period": "daily"}))
    if args.limit:
        symbols = symbols[: args.limit]
    print(f"[export] {len(symbols)} symbols to export")

    calendar = load_calendar(db)
    if not calendar:
        raise SystemExit("[export] empty calendar, aborting")
    cal_idx = {d: i for i, d in enumerate(calendar)}

    os.makedirs(os.path.join(args.out, "features"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "calendars"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "instruments"), exist_ok=True)
    with open(os.path.join(args.out, "calendars", "day.txt"), "w") as f:
        f.write("\n".join(calendar) + "\n")

    from tqdm import tqdm

    inst_lines, stats = [], {"exported": 0, "dropped_short": 0, "warn_ratio": 0, "total_ratio_forced": 0}
    for sym in tqdm(symbols, desc="export"):
        frame = build_symbol_frame(coll, sym)
        if frame is None:
            stats["dropped_short"] += 1
            continue

        # map onto the authoritative calendar between first/last valid day
        import bisect
        i0 = bisect.bisect_left(calendar, frame.index[0])
        i1 = bisect.bisect_right(calendar, frame.index[-1])
        sub_cal = calendar[i0:i1]
        if len(sub_cal) < MIN_ROWS:
            stats["dropped_short"] += 1
            continue
        aligned = frame.reindex(sub_cal)

        docs_probe = coll.find_one({"symbol": sym}, {"full_symbol": 1}) or {}
        prefix = symbol_prefix(sym, docs_probe.get("full_symbol") or "")
        inst = f"{prefix}{sym}"

        fdir = os.path.join(args.out, "features", inst)
        os.makedirs(fdir, exist_ok=True)
        first = cal_idx[sub_cal[0]]
        for field in FIELDS:
            if field == "factor":
                vals = np.ones(len(aligned), dtype=np.float32)
            else:
                vals = aligned[field].values.astype(np.float32)
            write_bin(os.path.join(fdir, f"{field}.day.bin"), first, vals)

        valid_days = aligned["close"].notna()
        start_day = sub_cal[int(np.argmax(valid_days.values))]
        end_day = sub_cal[len(sub_cal) - 1 - int(np.argmax(valid_days.values[::-1]))]
        inst_lines.append(f"{inst}\t{start_day}\t{end_day}")

        stats["exported"] += 1
        stats["total_ratio_forced"] += int(frame.attrs.get("n_ratio_forced", 0))
        if frame.attrs.get("n_ratio_forced", 0) > len(frame) * PRE_CLOSE_MISSING_LIMIT:
            stats["warn_ratio"] += 1

    bench = export_index(db, args.out, calendar)
    if bench:
        inst, b_start, b_end = bench
        inst_lines.append(f"{inst}\t{b_start}\t{b_end}")

    with open(os.path.join(args.out, "instruments", "all.txt"), "w") as f:
        f.write("\n".join(sorted(inst_lines)) + "\n")

    stats.update({
        "calendar_days": len(calendar),
        "calendar_range": [calendar[0], calendar[-1]],
        "elapsed_sec": round(time.time() - t0, 1),
        "benchmark": bench[0] if bench else None,
    })
    print(f"[export] done in {stats['elapsed_sec']}s: {stats}")
    with open(os.path.join(args.out, "export_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
def run_verify(args):
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D

    qlib.init(provider_uri=args.out, region=REG_CN)
    cal = D.calendar(freq="day")
    insts = D.list_instruments(D.instruments("all"), as_list=True)
    print(f"[verify] calendar days={len(cal)} range={cal[0].date()}..{cal[-1].date()}")
    print(f"[verify] instruments(all)={len(insts)}")

    rng = np.random.default_rng(42)
    sample = rng.choice(sorted(insts), size=min(6, len(insts)), replace=False)
    df = D.features(list(sample), ["$open", "$close", "$volume", "$vwap"], freq="day")
    print(f"[verify] sampled features shape={df.shape}")

    bad_ret, n_days_checked = 0, 0
    for inst in sample[:4]:
        closes = df.xs(inst, level=0)["$close"].dropna() if inst in df.index.get_level_values(0) else pd.Series(dtype=float)
        if len(closes) < 10:
            continue
        ret = closes.pct_change().dropna().abs()
        n_days_checked += len(ret)
        bad = ret[ret > 0.21]
        bad_ret += len(bad)
        for day, v in bad.head(5).items():
            print(f"[verify][WARN] {inst} {day.date()} |ret|={v:.3f}")
    print(f"[verify] |ret|>21% outliers: {bad_ret}/{n_days_checked} days "
          f"({100 * bad_ret / max(n_days_checked, 1):.3f}%)")
    bench = D.features(["SH000300"], ["$close"], freq="day")
    print(f"[verify] benchmark SH000300 rows={len(bench)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/workspace/data/qlib_data/cn_data_as")
    parser.add_argument("--uri", default=os.environ.get("MONGO_URI", DEFAULT_URI))
    parser.add_argument("--db", default=os.environ.get("MONGO_DB", DEFAULT_DB))
    parser.add_argument("--limit", type=int, default=0, help="export only first N symbols (smoke)")
    parser.add_argument("--verify", action="store_true", help="verify an existing dump")
    args = parser.parse_args()

    if args.verify:
        run_verify(args)
    else:
        run_export(args)


if __name__ == "__main__":
    sys.exit(main())
