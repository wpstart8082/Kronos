"""One-off probe: time auto_regressive_inference at several batch sizes.

Mimics QlibTestDataset windows exactly (per-window z-score on lookback only)
and reports ms/window + peak GPU memory for loader-equivalent batch sizes.
"""

import pickle
import sys
import time

import numpy as np
import torch

sys.path.append("/workspace")
from finetune.config import Config  # noqa: E402
from model.kronos import KronosTokenizer, Kronos, auto_regressive_inference  # noqa: E402

cfg = Config()
with open(f"{cfg.dataset_path}/test_data.pkl", "rb") as f:
    data = pickle.load(f)
syms = list(data.keys())[:6]

feats, tfeats = [], []
for s in syms:
    df = data[s].reset_index()
    df["minute"] = 0
    df["hour"] = 0
    df["weekday"] = df["datetime"].dt.weekday
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    n_win = min(60, len(df) - cfg.lookback_window - cfg.predict_window + 1)
    for i in range(max(n_win, 0)):  # up to 60 windows per symbol
        c = df.iloc[i : i + cfg.lookback_window]
        p = df.iloc[i + cfg.lookback_window : i + cfg.lookback_window + cfg.predict_window]
        x = c[cfg.feature_list].values.astype(np.float32)
        xs = c[cfg.time_feature_list].values.astype(np.float32)
        ys = p[cfg.time_feature_list].values.astype(np.float32)
        m, sd = np.mean(x, 0), np.std(x, 0)
        x = np.clip((x - m) / (sd + 1e-5), -cfg.clip, cfg.clip)
        feats.append(torch.from_numpy(x))
        tfeats.append((torch.from_numpy(xs), torch.from_numpy(ys)))

dev = torch.device("cuda:0")
tok = KronosTokenizer.from_pretrained(cfg.finetuned_tokenizer_path).to(dev).eval()
mdl = Kronos.from_pretrained(cfg.finetuned_predictor_path).to(dev).eval()

for ws in (200, 400, 1000):
    n = min(ws, len(feats))
    xb = torch.stack(feats[:n]).to(dev)
    xsb = torch.stack([t[0] for t in tfeats[:n]]).to(dev)
    ysb = torch.stack([t[1] for t in tfeats[:n]]).to(dev)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        auto_regressive_inference(
            tok, mdl, xb, xsb, ysb, max_context=cfg.max_context,
            pred_len=cfg.predict_window, clip=cfg.clip,
            T=0.6, top_k=0, top_p=0.9, sample_count=5,
        )
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"seq_batch={n}: {dt:.2f}s -> {dt / n * 1000:.0f} ms/window | "
          f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
