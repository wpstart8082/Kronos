# Kronos A股全市场微调训练计划（执行版）

> 本文档是实际执行的训练计划与运行记录。生成时间：2026-11-18。
> 执行环境：Windows + Docker GPU 容器（RTX 4090 D 24GB）。

## 1. 目标与结论摘要

用 `finetune/` 的 qlib 流水线，以本地 MongoDB 中全市场 A 股日 K 线微调
Kronos-Tokenizer-base + Kronos-base（102M 参数），并用 qlib TopkDropout 回测
（对比沪深300）验证微调是否提升预测效果。

**执行结果**（详见第 7 节）：

| 项目 | 结果 |
|---|---|
| 数据落盘 | 5761 只股票，2015-01-05 → 2026-11-17，后复权连续性校验 0 异常 |
| tokenizer 微调 | 30 epochs / 37 分钟，val loss 0.0096 → **0.0078**（-19%） |
| predictor 微调 | 30 epochs / 2h20m，best val loss **3.1466**（随机基线 ≈13.9） |
| 回测（微调后，`max` 信号） | 含成本超额 **+4.82%/年**，IR 0.28，超额回撤 -18.7% |
| 回测（零样本基线，`max` 信号） | 含成本超额 **-14.18%/年**，IR -0.80，回撤 -51.5% |
| **微调净效应（`max`）** | **约 +19 个百分点/年的超额收益改善**，四个信号全部同向改善 |

核心结论：在 2025-07 → 2026-11 全市场 Top50 回测中，零样本 Kronos-base 的选股信号
全部显著为负；经全市场日 K 微调后，`max`/`min` 信号翻正（最强 +4.82%/年含成本），
回撤显著收窄——微调对 A 股预测有效性的提升得到直接量化验证。

## 2. 数据源与审计（实际值）

MongoDB：`tradingagentscn.stock_daily_quotes`（容器 `tradingagents-mongodb`，
通过 docker 网络 `tradingagents-cn_tradingagents-network` 以主机名 `mongodb` 访问）。

- 全量日 K：17,726,960 条（period=daily），1990-12-19 → 2026-11-17，5922 只
- 其中 **未复权新数据（adjust="none"，tushare）**：11,946,589 条，
  2015-01-05 → 2026-11-17，5838 只，pre_close 缺失仅 227 条（0.002%）
- 老 qfq 快照（无 adjust 字段）：约 577 万条 —— **全部弃用**（与未复权数据拼接会产生价格断层）
- 基准指数：`index_daily_quotes` 中 000300（symbol=000300.SH）6032 条，
  2002-01-04 起（早期 open/high/low 为 null，以 close 回填）

审计报告留档：`data/audit/mongo_audit_report.json`（容器 volume 内）。

## 3. 数据准备流水线

### 3.1 导出与复权（`finetune/data_prep/export_mongo_to_qlib.py`）

去重规则（复刻 TradingAgents-CN）：同日多源时 `adjust=="none"` 优先，再按
tushare > akshare > baostock；仅保留未复权文档，qfq-only 的日期留空（NaN）。

后复权：逐股按日排序，`step_t = close_{t-1} / pre_close_t`（缺失/非法置 1），
`F = cumprod(step)`，`adj_OHLC = 原始 OHLC × F`。
数学性质：`adj_close_t / adj_close_{t-1} = close_t / pre_close_t`（真实日收益，
含除权日与停牌跨缺口）。**注意方向**：step 是 `prev_close/pre_close` 而非其倒数
（开发中曾因方向写反产生 6 个 >21% 的假异常收益，修复后 0 异常——此校验
建议每次导出后必跑）。

qlib 落盘格式：`calendars/day.txt`（3129 个交易日，2014-01-02 起）、
`instruments/all.txt`（`sh600000\t起\t止`）、
`features/<inst>/<field>.day.bin`（小端 float32，首元素=日历起始索引；
**文件名必须带 `.day` 频次后缀**）。停牌日 reindex 后填 NaN。
字段：open/close/high/low（后复权）、volume（股数）、vwap（amount/volume×F，
回退 close）、factor（恒 1）。

### 3.2 实际产出

- 导出耗时 77 秒（容器间网络），5761 只入池、161 只因不足 101 个交易日剔除
- `--verify` 回读校验：instruments 5762（含 sh000300）、抽样特征可读、
  \|日收益\|>21% 异常点 **0/8684**
- 预处理（`finetune/qlib_data_preprocess.py`）约 3 分钟：
  train 252MB / val 71MB / test 75MB（pickle），**训练可用窗口 7,644,922 个**

## 4. 训练配置（finetune/config.py 实际值）

| 配置 | 值 | 说明 |
|---|---|---|
| instrument | `all`（全市场，基准 SH000300） | |
| dataset 区间 | 2015-01-01 → 2026-11-17 | |
| train | 2015-01-01 → 2023-12-31 | 9 年 |
| val | 2023-10-01 → 2025-06-30 | 与 train 重叠吸收 lookback |
| test | 2025-02-01 → 2026-11-17 | 比 backtest 提前 ~100 交易日（lookback 90+predict 10 的窗口缓冲） |
| backtest | 2025-07-01 → 2026-11-17 | ~17 个月 |
| lookback / predict / max_context | 90 / 10 / 512 | 上游默认 |
| epochs / batch_size | 30 / 50 | 上游默认；4090 实测显存 ~4.5GB（tokenizer）/（待测 predictor） |
| n_train_iter / n_val_iter | 100,000 / 20,000（每 epoch 采样） | 上游默认 2000×bs / 400×bs |
| tokenizer_lr / predictor_lr | 2e-4 / 4e-5 | 上游默认 |
| 优化器 | AdamW(0.9, 0.95, wd=0.1) + OneCycleLR(pct_start 0.03) | 上游默认 |
| 早停 | 无（每 epoch 验证，保留 val loss 最优 checkpoint） | 上游设计 |
| 底座 | `models/Kronos-Tokenizer-base` + `models/Kronos-base`（本地） | |
| use_comet | False | comet_ml 仍需安装（脚本顶层 import） |

不改动的文件：`train_tokenizer.py`、`train_predictor.py`、`dataset.py`
（已含 #227 防泄漏归一化）、`model/`。
`qlib_test.py` 仅加 2 行：`FT_TOKENIZER_PATH`/`FT_MODEL_PATH` 环境变量覆盖
（用于零样本基线）。

## 5. 运行环境

- 镜像 `kronos-finetune:latest`：`pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`
  + pandas 2.2.2 + pyqlib 0.9.7 + comet_ml + pymongo 等（Dockerfile：
  `docker/finetune.Dockerfile`，pip 走清华镜像——PyPI 直连在本网络下极慢）
- 容器 `kronos-finetune`：`--gpus all`、加入 `tradingagents-cn_tradingagents-network`、
  代码 bind mount `E:/AIProject/Kronos:/workspace`、
  数据/产物走 named volume（`kronos-finetune-data`、`kronos-finetune-outputs`）
- **启动工作目录必须是 `/workspace/finetune`**：脚本内 `sys.path.append("../")`
  是 CWD 相对路径，从仓库根启动会报 `No module named 'model'`

## 6. 复跑命令

```bash
# 一次性环境（已完成）：
docker build -f docker/finetune.Dockerfile -t kronos-finetune:latest docker
MSYS_NO_PATHCONV=1 docker run -d --name kronos-finetune --gpus all \
  --network tradingagents-cn_tradingagents-network \
  -v "E:/AIProject/Kronos:/workspace" \
  --mount source=kronos-finetune-data,target=/workspace/data \
  --mount source=kronos-finetune-outputs,target=/workspace/outputs \
  kronos-finetune:latest sleep infinity

# 各阶段（容器内，支持断点续跑，日志在 /workspace/outputs/logs/）：
docker exec kronos-finetune bash -c "cd /workspace && bash finetune/run_finetune.sh audit"
docker exec kronos-finetune bash -c "cd /workspace && bash finetune/run_finetune.sh export"
docker exec kronos-finetune bash -c "cd /workspace && bash finetune/run_finetune.sh verify"
docker exec kronos-finetune bash -c "cd /workspace && bash finetune/run_finetune.sh preprocess"
docker exec kronos-finetune bash -c "cd /workspace && bash finetune/run_finetune.sh selfcheck"
docker exec kronos-finetune bash -c "cd /workspace && bash finetune/run_finetune.sh smoke"
docker exec kronos-finetune bash -c "cd /workspace && bash finetune/run_finetune.sh tokenizer"
docker exec kronos-finetune bash -c "cd /workspace && bash finetune/run_finetune.sh predictor"
docker exec kronos-finetune bash -c "cd /workspace && bash finetune/run_finetune.sh backtest"
# 复用已生成的 predictions.pkl 只重跑组合回测（分钟级）：
docker exec kronos-finetune bash -c "cd /workspace/finetune && python backtest_only.py /workspace/outputs/backtest_results/as_backtest_v1/predictions.pkl"
docker exec kronos-finetune bash -c "cd /workspace && bash finetune/run_finetune.sh baseline"
```

进度监控：`docker exec kronos-finetune tail -f /workspace/outputs/logs/train_tokenizer.log`
（每 100 step 打印 loss/LR，每 epoch 打印验证摘要与最优 checkpoint）。

## 7. 训练与回测结果（执行后回填）

### 7.1 tokenizer 微调

- 30 epochs × 2000 steps（每 epoch 采样 10 万窗口），总耗时 **37 分钟**（RTX 4090 D，batch 50，显存 ~4.5GB）
- 验证 loss：0.0096 → 单调下降 → **0.0078**（最优，best checkpoint 自动保存）
- 产物：`/workspace/outputs/models/as_tokenizer_v1/checkpoints/best_model`（volume 内）

### 7.2 predictor 微调

- 30 epochs × 2000 steps，总耗时 **2h20m**（每 epoch 4分43秒，显存 ~10.3GB，GPU 利用率 97%）
- 训练 loss：3.2 → 2.7；验证 loss 最优 **3.1466**（逐 epoch 保存最优；对照：token 均匀随机基线 ≈ 2×10×ln2 ≈ 13.9，微调后 tokenizer 的 token 空间高度可预测）
- 产物：`/workspace/outputs/models/as_predictor_v1/checkpoints/best_model`

### 7.2b 推理性能调优（回测前必读）

全市场推理原始配置（fp32、采样5）实测外推需 **~19 小时**/轮，不可行。分量计时
（`finetune/data_prep/probe_components.py`，batch 1000×seq90）：
`decode_s1` 0.73s（绝对主导，每批 10 步自回归）≫ encode/decode_s2/decode（<0.05s）。

采用的优化（不改模型结构）：
1. `qlib_test.py` 推理调用包 `torch.autocast(bfloat16)`——`decode_s1` 0.73→0.38s
2. `inference_sample_count` 5 → 3（信号为多路径均值，方差影响轻微）
3. `model/kronos.py` 的 `z.cpu().numpy()` 改 `z.cpu().float().numpy()`（bf16 转 numpy 的必要上转型，fp32 下为空操作）

最终 **3.7s/迭代 × 5451 迭代 ≈ 5.6 小时/轮**（显存 12.9GB）。

### 7.3 回测（微调后，2025-07-01 → 2026-11-10，Topk=50/Drop=5/持有≥5日，T+1开盘，对比沪深300）

四个信号的年化超额收益（`backtest_only.py` 复用全市场推理产物）：

| 信号 | 超额年化（无成本） | 超额年化（含成本） | IR（含成本） | 超额最大回撤 |
|---|---|---|---|---|
| `last`（末日预测涨跌） | +3.05% | -0.24% | -0.01 | -27.9% |
| `mean`（10日均值涨跌） | +1.39% | -1.69% | -0.09 | -17.4% |
| **`max`（10日上沿涨跌）** | **+8.61%** | **+4.82%** | **0.28** | **-18.7%** |
| `min`（10日下沿涨跌） | +4.62% | +1.73% | 0.10 | -22.8% |

同期基准沪深300：年化 +3.61%、最大回撤 -23.1%。最强信号 `max`（预测10日最高价上沿）
绝对年化 ≈ 8.4%，回撤 -18.7%——收益与回撤均优于持有基准。交易成本假设：开仓 0.1%、
平仓 0.15%、最低 5 元。

注意事项：
- 回测 `end_time` 不能等于日历最后一天（qlib `calendar[index+1]` 越界），已改为 2026-11-10；
- 零样本基线用 `FT_RESULT_NAME=as_baseline_v1` 隔离产物，避免覆盖微调版 predictions.pkl；
- 净值曲线图：容器内 `../figures/backtest_result_example.png`（相对 finetune/ 目录）。

### 7.4 零样本基线 vs 微调（同回测配置：2025-07-01 → 2026-11-10，Topk=50/Drop=5）

年化超额收益（含成本 / 无成本），IR 为含成本口径：

| 信号 | 零样本（含成本） | 微调后（含成本） | 微调净改善 | 微调 IR |
|---|---|---|---|---|
| `last` | -13.82% | -0.24% | **+13.6pp** | -0.01 |
| `mean` | -10.06% | -1.69% | **+8.4pp** | -0.09 |
| **`max`** | **-14.18%** | **+4.82%** | **+19.0pp** | **0.28** |
| `min` | -11.00% | +1.73% | **+12.7pp** | 0.10 |

零样本超额最大回撤 -39% ~ -52%，微调后收窄至 -17% ~ -28%。

**解读**：
- 零样本 Kronos（在全球 K 线上预训练、未见过 A 股涨跌停/散户结构）在全市场选股上
  信号系统性为负——直接用于 A 股不可行；
- 微调后模型学到了 A 股分布：四个信号全部大幅改善，`max`（预测 10 日价格上沿）
  含成本超额 +4.82%/年、绝对年化 ≈8.4%、回撤优于基准（-18.7% vs -23.1%）；
- 本回测为上游 demo 口径（无行业/风格中性化、无组合优化、单边成本 0.1%/0.15%、
  涨跌停 0.095 过滤、T+1 开盘成交），实盘前需叠加更严谨的模拟。

产物（宿主机 `E:\AIProject\Kronos\outputs\`）：
- `models/as_tokenizer_v1/`、`models/as_predictor_v1/`（微调后 checkpoint，可直接加载）
- `backtest_results/predictions_finetuned.pkl`（微调版全市场信号）
- `backtest_results/finetuned_curves.png` / `baseline_curves.png`（净值曲线）
- `backtest_results/backtest_finetuned_report.log` / `baseline_report.log`（完整报告）
- 容器 volume 内另有 `as_baseline_v1/predictions.pkl`（零样本信号）

## 8. 已知坑与经验

1. **PyPI 直连慢**：本网络下 pip 安装 30 分钟不完，换清华镜像 60-90 秒完成。
2. **qlib bin 文件名**：必须 `<field>.<freq>.bin`（如 `open.day.bin`），
   写成 `open.bin` 时 qlib 静默返回空 DataFrame。
3. **复权比率方向**：`prev_close/pre_close`，写反后除权日会出现 ±60% 假收益，
   `--verify` 的 |ret|>21% 检查能兜底。
4. **启动目录**：train 脚本须在 `finetune/` 内启动（相对 sys.path）。
5. Git Bash 下 docker `--mount target=/...` 会被路径改写，加 `MSYS_NO_PATHCONV=1`。
6. Windows 下 nccl 不可用：选择 Docker GPU 容器后训练代码零改动。

## 9. 部署：kronos-api 原版与 A 股微调版并存（2026-11-19 完成）

- 模型目录（原版未动）：
  `models/Kronos-AS-v1/`（微调预测器 391MB）+
  `models/Kronos-Tokenizer-AS-v1/`（微调分词器 16MB）
- `kronos_service/app/model_manager.py` REGISTRY 新增
  `"kronos-as-base"`（display_name「Kronos-base·A股微调v1」，**hf_id 留空**——
  目录缺失时宁可报错也绝不静默回退到原版权重冒充微调版）
- 镜像重建：`cd kronos_service && docker compose build && docker compose up -d`
- 使用：`POST /v1/predict` 的 `model` 字段传 `kronos-as-base`（原模型继续用
  `kronos-mini/small/base`）；A 股日 K 建议参数 `pred_len=10, temperature=0.6,
  top_p=0.9, sample_count=3`（与回测一致），输入用**后复权**价格
- 验证记录（`outputs/deploy_verify/`）：600519 后复权 90 日输入，
  两模型 200 OK 且预测方向相反（原版看空 -9.5%、微调版看平略多 +1.1%），
  双模型同时在 GPU 上服务正常
- 验证用 API key 明文存于 `outputs/deploy_verify/.verify_api_key`（kronos-admin
  可随时吊销）
