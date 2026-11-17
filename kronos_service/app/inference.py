"""Inference service layer: request validation, preparation, prediction."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .model_manager import REGISTRY
from .timestamps import TimestampError, build_future_timestamps, infer_frequency

_REQUIRED_COLS = ["open", "high", "low", "close"]


class DataValidationError(ValueError):
    """Client data problem -> HTTP 422."""


@dataclass
class PredictParams:
    pred_len: int
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.9
    sample_count: int = 1

    def signature(self) -> tuple:
        return (
            self.pred_len,
            round(self.temperature, 6),
            self.top_k,
            round(self.top_p, 6),
            self.sample_count,
        )


@dataclass
class PreparedInput:
    df: pd.DataFrame
    x_timestamp: pd.Series
    y_timestamp: pd.Series
    lookback: int
    pred_len: int
    freq: str
    model_key: str
    symbol: str | None = None
    input_from: Any = None
    input_to: Any = None

    def batch_signature(self, params: PredictParams) -> tuple:
        """Signature for dynamic batching (same shape + same sampling)."""
        return (self.model_key, self.lookback, self.pred_len, params.signature())


def parse_candles(
    data: list[dict[str, Any]],
    model_key: str,
    cfg,
    lookback_requested: int | None,
    pred_len: int,
    future_timestamps: list | None,
    symbol: str | None = None,
) -> PreparedInput:
    """Validate raw request candles and build the model input.

    ``cfg`` is the RuntimeConfig instance providing limit values.
    """
    max_rows = cfg.get_int("max_data_rows")
    if not data:
        raise DataValidationError("data 不能为空")
    if len(data) > max_rows:
        raise DataValidationError(f"数据行数 {len(data)} 超过上限 {max_rows}")

    rows = []
    for i, item in enumerate(data):
        try:
            rows.append(
                {
                    "timestamps": pd.to_datetime(item["timestamp"], errors="raise"),
                    "open": float(item["open"]),
                    "high": float(item["high"]),
                    "low": float(item["low"]),
                    "close": float(item["close"]),
                    "volume": float(item["volume"]) if item.get("volume") is not None else 0.0,
                    "amount": float(item["amount"]) if item.get("amount") is not None else 0.0,
                }
            )
        except (KeyError, TypeError, ValueError) as e:
            raise DataValidationError(f"第 {i} 行数据无效: {e}") from e

    df = pd.DataFrame(rows)
    for col in _REQUIRED_COLS + ["volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=_REQUIRED_COLS + ["volume", "amount"])
    dropped = before - len(df)
    if dropped:
        # NaN rows are dropped (mirrors webui behaviour) but too many is suspicious
        if dropped > max(10, before // 10):
            raise DataValidationError(f"包含无效数值(NaN)的行过多: {dropped}/{before}")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataValidationError("价格数据包含非正数，请检查 open/high/low/close")

    df = df.sort_values("timestamps").drop_duplicates(subset="timestamps").reset_index(drop=True)

    min_lookback = cfg.get_int("min_lookback")
    max_lookback = min(cfg.get_int("max_lookback"), REGISTRY[model_key]["max_context"])
    default_lookback = min(cfg.get_int("default_lookback"), max_lookback)
    lookback = lookback_requested or default_lookback
    lookback = max(min_lookback, min(lookback, max_lookback))

    if len(df) < max(min_lookback, 2):
        raise DataValidationError(f"有效数据不足: 需要 >= {min_lookback} 根, 实际 {len(df)} 根")
    if len(df) < lookback:
        lookback = len(df)  # use whatever history is available, >= min_lookback

    df = df.tail(lookback).reset_index(drop=True)
    x_timestamp = df["timestamps"]
    try:
        y_timestamp = build_future_timestamps(x_timestamp, pred_len, future_timestamps)
    except TimestampError as e:
        raise DataValidationError(str(e)) from e

    freq = infer_frequency(x_timestamp)
    return PreparedInput(
        df=df[_REQUIRED_COLS + ["volume", "amount"]],
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        lookback=lookback,
        pred_len=pred_len,
        freq=freq,
        model_key=model_key,
        symbol=symbol,
        input_from=x_timestamp.iloc[0],
        input_to=x_timestamp.iloc[-1],
    )


def run_prediction(predictor, prepared: PreparedInput, params: PredictParams) -> tuple[pd.DataFrame, int]:
    """Run a single prediction; returns (pred_df, inference_ms)."""
    t0 = time.perf_counter()
    pred_df = predictor.predict(
        df=prepared.df,
        x_timestamp=prepared.x_timestamp,
        y_timestamp=prepared.y_timestamp,
        pred_len=params.pred_len,
        T=params.temperature,
        top_k=params.top_k,
        top_p=params.top_p,
        sample_count=params.sample_count,
        verbose=False,
    )
    return pred_df, int((time.perf_counter() - t0) * 1000)


def run_prediction_batch(
    predictor, prepared_list: list[PreparedInput], params: PredictParams
) -> tuple[list[pd.DataFrame], int]:
    """Batched prediction (all items must share lookback/pred_len)."""
    t0 = time.perf_counter()
    pred_dfs = predictor.predict_batch(
        [p.df for p in prepared_list],
        [p.x_timestamp for p in prepared_list],
        [p.y_timestamp for p in prepared_list],
        pred_len=params.pred_len,
        T=params.temperature,
        top_k=params.top_k,
        top_p=params.top_p,
        sample_count=params.sample_count,
        verbose=False,
    )
    return pred_dfs, int((time.perf_counter() - t0) * 1000)


def format_predictions(pred_df: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for ts, row in pred_df.iterrows():
        out.append(
            {
                "timestamp": pd.Timestamp(ts).isoformat(),
                "open": round(float(row["open"]), 6),
                "high": round(float(row["high"]), 6),
                "low": round(float(row["low"]), 6),
                "close": round(float(row["close"]), 6),
                "volume": round(float(row["volume"]), 2),
                "amount": round(float(row["amount"]), 2),
            }
        )
    return out


def check_prediction_sane(pred_df: pd.DataFrame, pred_len: int) -> None:
    if len(pred_df) != pred_len:
        raise RuntimeError(f"预测输出长度异常: {len(pred_df)} != {pred_len}")
    if pred_df.isna().any().any():
        raise RuntimeError("预测输出包含 NaN")
