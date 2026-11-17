"""Frequency inference and future timestamp generation."""
from __future__ import annotations

import pandas as pd
import pytest

from app.timestamps import build_future_timestamps, infer_frequency


def _daily_series(n=30, start="2026-10-01"):
    return pd.Series(pd.bdate_range(start=start, periods=n))


def _intraday_series(days=3):
    times = ["09:35", "09:40", "09:45", "10:00", "10:05"]
    out = []
    base = pd.Timestamp("2026-11-11")  # Wednesday
    for d in range(days):
        day = base + pd.Timedelta(days=d)
        for t in times:
            out.append(day + pd.Timedelta(hours=int(t[:2]), minutes=int(t[3:])))
    return pd.Series(pd.to_datetime(out))


def test_infer_frequency_daily():
    assert infer_frequency(_daily_series()) == "daily"
    assert infer_frequency(_intraday_series()) == "intraday"


def test_future_daily_skips_weekend():
    ts = _daily_series(10, start="2026-11-02")  # Mon..Fri(6), Mon..Fri(10) ends Fri 2026-11-13
    future = build_future_timestamps(ts, 5)
    assert len(future) == 5
    weekdays = {t.weekday() for t in future}
    assert weekdays <= {0, 1, 2, 3, 4}
    assert (future.iloc[0] - ts.iloc[-1]).days >= 3  # Fri -> next Mon at least


def test_future_intraday_replays_session_grid():
    ts = _intraday_series(3)  # ends Fri 2026-11-13 10:05
    future = build_future_timestamps(ts, 7)
    assert len(future) == 7
    assert future.iloc[0].weekday() == 0  # next Monday
    assert future.iloc[0].hour == 9 and future.iloc[0].minute == 35
    times = [(t.hour, t.minute) for t in future]
    assert times[:5] == [(9, 35), (9, 40), (9, 45), (10, 0), (10, 5)]
    assert times[5] == (9, 35)  # next day repeats grid


def test_explicit_future_timestamps_override():
    ts = _daily_series(5)
    explicit = ["2026-12-01 00:00:00", "2026-12-02 00:00:00"]
    future = build_future_timestamps(ts, 2, explicit=explicit)
    assert list(future) == list(pd.to_datetime(explicit))


def test_explicit_length_mismatch_raises():
    with pytest.raises(Exception):
        build_future_timestamps(_daily_series(5), 3, explicit=["2026-12-01"])
