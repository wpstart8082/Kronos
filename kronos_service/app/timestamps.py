"""Frequency inference and future timestamp generation.

KronosPredictor needs calendar features (minute/hour/weekday/day/month) for
the prediction horizon, so the server must synthesize the future timestamps
from the client's history when the request does not provide them explicitly.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

# Seconds between consecutive bars -> frequency label
_DAILY_THRESHOLD = 12 * 3600  # >= 12h gap treated as daily-or-coarser bars


class TimestampError(ValueError):
    pass


def infer_frequency(ts: pd.Series) -> str:
    """Classify history cadence: 'daily' | 'intraday' | 'generic'."""
    if len(ts) < 3:
        return "generic"
    deltas = ts.diff().dropna()
    median_seconds = deltas.dt.total_seconds().median()
    if median_seconds >= _DAILY_THRESHOLD:
        return "daily"
    return "intraday"


def build_future_timestamps(
    x_timestamp: pd.Series, pred_len: int, explicit: list | None = None
) -> pd.Series:
    """Return ``pred_len`` future timestamps following the history.

    - explicit: caller-provided values (parsed & length-checked)
    - daily bars: business days, preserving time-of-day
    - intraday bars: reuse the historical intraday session grid,
      skipping weekends (holidays are not known offline - callers who need
      exchange-calendar precision should pass future_timestamps explicitly)
    - fallback: extrapolate the median delta
    """
    ts = pd.Series(pd.to_datetime(x_timestamp)).sort_values().reset_index(drop=True)
    if explicit is not None:
        if len(explicit) != pred_len:
            raise TimestampError(f"future_timestamps 长度({len(explicit)})必须等于 pred_len({pred_len})")
        parsed = pd.to_datetime(list(explicit), errors="coerce")
        if parsed.isna().any():
            raise TimestampError("future_timestamps 中存在无法解析的时间戳")
        return pd.Series(parsed, name="timestamps")

    if len(ts) < 2:
        raise TimestampError("历史数据不足 2 根，无法推断频率；请显式传入 future_timestamps")

    freq = infer_frequency(ts)
    if freq == "daily":
        return _future_daily(ts, pred_len)
    if freq == "intraday":
        try:
            return _future_intraday(ts, pred_len)
        except TimestampError:
            pass  # fall through to generic
    return _future_generic(ts, pred_len)


def _future_daily(ts: pd.Series, pred_len: int) -> pd.Series:
    last = ts.iloc[-1]
    time_of_day = last.time()
    # bdate_range gives business days after the last bar
    days = pd.bdate_range(start=last + timedelta(days=1), periods=pred_len)
    values = [datetime.combine(d.date(), time_of_day) for d in days]
    return pd.Series(pd.to_datetime(values), name="timestamps")


def _future_intraday(ts: pd.Series, pred_len: int) -> pd.Series:
    """Replay the historical intraday session grid on following weekdays."""
    last_days = ts.dt.normalize().drop_duplicates().tail(7)
    if len(last_days) < 2:
        raise TimestampError("日内频率推断需要至少 2 个交易日")
    # collect the unique time-of-day grid from recent days (ordered)
    grid_times: list = []
    seen = set()
    for day in last_days:
        day_times = ts[ts.dt.normalize() == day].dt.time.tolist()
        for t in day_times:
            if t not in seen:
                seen.add(t)
                grid_times.append(t)
    if not grid_times:
        raise TimestampError("无法从历史数据提取日内时间网格")
    grid_times.sort()

    out: list[datetime] = []
    last = ts.iloc[-1]
    day = last.normalize()
    # advance from the bar right after the last one
    while len(out) < pred_len:
        day = day + timedelta(days=1)
        if day.weekday() >= 5:  # Sat/Sun
            continue
        for t in grid_times:
            candidate = datetime.combine(day.date(), t)
            if candidate > last.to_pydatetime():
                out.append(candidate)
                if len(out) >= pred_len:
                    break
    return pd.Series(pd.to_datetime(out), name="timestamps")


def _future_generic(ts: pd.Series, pred_len: int) -> pd.Series:
    deltas = ts.diff().dropna()
    delta = deltas.median()
    if pd.isna(delta) or delta <= timedelta(0):
        delta = timedelta(days=1)
    last = ts.iloc[-1]
    out = [last + delta * (i + 1) for i in range(pred_len)]
    return pd.Series(pd.to_datetime(out), name="timestamps")
