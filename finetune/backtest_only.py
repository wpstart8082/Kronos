"""Backtest-only runner: reuse a saved predictions.pkl and skip inference.

The full qlib_test.py takes ~5.5h for inference on the full market; when the
predictions are already computed only the portfolio backtest needs to rerun
(e.g. after fixing backtest_time_range).

Usage (inside the container, from finetune/):
    python backtest_only.py /workspace/outputs/backtest_results/as_backtest_v1/predictions.pkl
"""

import os
import pickle
import sys

from config import Config
from qlib_test import QlibBacktest


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    pred_path = sys.argv[1]

    with open(pred_path, "rb") as f:
        model_preds = pickle.load(f)
    print(f"loaded predictions: {list(model_preds.keys())}, "
          f"{model_preds[list(model_preds.keys())[0]].shape}")

    backtester = QlibBacktest(Config())
    backtester.run_and_plot_results(model_preds)


if __name__ == "__main__":
    main()
