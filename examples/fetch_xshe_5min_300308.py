# -*- coding: utf-8 -*-
"""
获取中际旭创(300308.SZ)5分钟K线，保存为 Kronos predict 所需格式:
  data/XSHE_5min_300308.csv
列: timestamps, open, high, low, close, volume, amount
单位: 价格=元, volume=手(东财口径), amount=元
"""
import os
import akshare as ak
import pandas as pd

SYMBOL = "300308"
MARKET = "XSHE"  # 深交所
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT_PATH = os.path.join(OUT_DIR, f"{MARKET}_5min_{SYMBOL}.csv")

# 时间范围: 取约2个月历史，保证 >= lookback 400 + pred_len 120 = 520 根
START = "2026-09-14 09:30:00"
END = "2026-11-16 15:00:00"

df = ak.stock_zh_a_hist_min_em(
    symbol=SYMBOL,
    period="5",
    start_date=START,
    end_date=END,
    adjust="",  # 不复权，保持原始价格
)

# 东财列名 -> Kronos 列名（注意东财顺序: 时间/开盘/收盘/最高/最低/...）
rename_map = {
    "时间": "timestamps",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}
df = df.rename(columns=rename_map)
df["timestamps"] = pd.to_datetime(df["timestamps"])
df = df.sort_values("timestamps").reset_index(drop=True)
df = df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]

os.makedirs(OUT_DIR, exist_ok=True)
df.to_csv(OUT_PATH, index=False)

# ---- 校验 ----
print(f"保存至: {OUT_PATH}")
print(f"总行数: {len(df)} (需 >= 520)")
print(f"时间范围: {df['timestamps'].iloc[0]} ~ {df['timestamps'].iloc[-1]}")
print(f"缺失值: {df.isna().sum().sum()}")
per_day = df.groupby(df["timestamps"].dt.date).size()
print(f"交易日数: {len(per_day)}, 每日K线数: min={per_day.min()}, max={per_day.max()}")
print(f"时间戳单调递增: {df['timestamps'].is_monotonic_increasing}")
print("\n前3行:")
print(df.head(3).to_string(index=False))
print("\n后3行:")
print(df.tail(3).to_string(index=False))
