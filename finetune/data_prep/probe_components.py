"""Micro-benchmark: time each inference component at batch ~600, fp32 vs bf16.

Pins down which part of auto_regressive_inference dominates and quantifies the
autocast speedup before we commit to a full backtest run.
"""

import pickle
import sys
import time

import numpy as np
import torch

sys.path.append("/workspace")
from finetune.config import Config  # noqa: E402
from model.kronos import KronosTokenizer, Kronos  # noqa: E402

cfg = Config()
with open(f"{cfg.dataset_path}/test_data.pkl", "rb") as f:
    data = pickle.load(f)
sym = next(s for s in data if len(data[s]) > 200)
df = data[sym].reset_index()
df["minute"] = 0
df["hour"] = 0
df["weekday"] = df["datetime"].dt.weekday
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month

N = 200  # windows -> x5 sample_count = 1000 inner sequences
xs_list, ts_list = [], []
for i in range(N):
    c = df.iloc[i : i + cfg.lookback_window]
    x = c[cfg.feature_list].values.astype(np.float32)
    xs = c[cfg.time_feature_list].values.astype(np.float32)
    m, sd = np.mean(x, 0), np.std(x, 0)
    xs_list.append(torch.from_numpy(np.clip((x - m) / (sd + 1e-5), -cfg.clip, cfg.clip)))
    ts_list.append(torch.from_numpy(xs))

dev = torch.device("cuda:0")
tok = KronosTokenizer.from_pretrained(cfg.finetuned_tokenizer_path).to(dev).eval()
mdl = Kronos.from_pretrained(cfg.finetuned_predictor_path).to(dev).eval()

x = torch.stack(xs_list).to(dev)
x_stamp = torch.stack(ts_list).to(dev)
x5 = x.unsqueeze(1).repeat(1, 5, 1, 1).reshape(-1, x.size(1), x.size(2))
s5 = x_stamp.unsqueeze(1).repeat(1, 5, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2))
print(f"inner batch={x5.size(0)} seq={x5.size(1)}")

def bench(name, fn, reps=3):
    fn()  # warmup
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    print(f"{name:28s} {(time.time()-t0)/reps:8.2f} s")

for amp in (False, True):
    ctx = torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp)
    tag = "bf16" if amp else "fp32"
    print(f"--- {tag} ---")
    with torch.no_grad(), ctx:
        bench(f"[{tag}] tokenizer.encode", lambda: tok.encode(x5, half=True))
        toks = tok.encode(x5, half=True)
        stamp = s5
        bench(f"[{tag}] model.decode_s1", lambda: mdl.decode_s1(toks[0], toks[1], stamp))
        s1_logits, context = mdl.decode_s1(toks[0], toks[1], stamp)
        s1_last = s1_logits[:, -1, :].float()
        sample_pre = torch.multinomial(torch.softmax(s1_last / 0.6, -1), 1)
        bench(f"[{tag}] model.decode_s2", lambda: mdl.decode_s2(context, sample_pre))
        ztok = [toks[0], toks[1]]
        bench(f"[{tag}] tokenizer.decode", lambda: tok.decode(ztok, half=True))
