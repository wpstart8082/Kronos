# Kronos 预测服务（API + 管理后台 + Docker 部署）

将本仓库的 Kronos K 线预测模型封装为生产级 REST 服务：调用方携带 K 线数据即可获得股票/指数等任意品种的未来走势预测；配套管理后台完成密钥签发、参数配置、监控与自检。

```
调用方 ──X-API-Key──> FastAPI :8000
   ├─ /v1/predict        同步预测（排队/推理耗时分解）
   ├─ /v1/predict/async  异步预测（202 + task_id）
   ├─ /v1/tasks/{id}     任务查询
   ├─ /admin             管理后台（Vue3 + Element Plus + ECharts）
   ├─ /api/docs          OpenAPI 交互文档
   └─ /healthz /readyz   探活（供 Docker/监控）
```

- 单容器单进程：模型常驻内存，推理经有界队列 + 信号量并发控制，GPU（宿主机直通）上 pred_len=120 约秒级返回
- **双模型自动路由**：未指定 `model` 时按标的类型分流 —— 指数 → `kronos-base`（原版零样本，指数回测最优），股票 → `kronos-as-base`（A股全市场微调版）；两个模型常驻显存（合计约 0.9GB），无需切换等待
- 模型权重挂载宿主机 `./models/`（本地优先，缺失走 HF Hub 下载，同样落在宿主机）
- SQLite(WAL) 存密钥/请求日志/任务/运行配置；日志双写控制台与 `logs/` 滚动文件

---

## 1. 准备模型权重

线上仅部署两个模型（双模型路由的两侧），`kronos-mini`/`kronos-small` 已于 2026-11 清理退役（HF 原版 `NeoQuasar/Kronos-*` 仍可随时取回）：

```bash
# ① 下载 kronos-base + 配套 Tokenizer（优先从本机 HF 缓存拷贝，缺失才联网）
python kronos_service/scripts/download_models.py
```

生成结构（`config.json` + `*.safetensors`，均为 `from_pretrained` 可直接读取的本地目录）：

```
models/
├── Kronos-base/  └── Kronos-Tokenizer-base/
```

**② A股微调版（`kronos-as-base`，本地权重、不走 HF）**：从训练容器卷导出并改名为注册表要求的目录名（Git Bash 下路径参数前加 `MSYS_NO_PATHCONV=1`）：

```bash
MSYS_NO_PATHCONV=1 docker cp kronos-finetune:/workspace/outputs/models/as_predictor_v1  models/Kronos-AS-v1
MSYS_NO_PATHCONV=1 docker cp kronos-finetune:/workspace/outputs/models/as_tokenizer_v1  models/Kronos-Tokenizer-AS-v1
```

| 注册名 | 权重目录 | 用途 |
|---|---|---|
| `kronos-base` | `Kronos-base` + `Kronos-Tokenizer-base` | 原版零样本；**指数路由目标**（指数回测最优 mean 信号） |
| `kronos-as-base` | `Kronos-AS-v1` + `Kronos-Tokenizer-AS-v1` | A股全市场微调v1；**股票路由目标**（股票回测大幅改善） |

## 2. Docker 部署（GPU）

宿主机要求（Windows）：NVIDIA 驱动（≥531）+ Docker Desktop（WSL2 后端）。**无需安装 CUDA Toolkit**——镜像自带 CUDA 运行时，compose 的 `gpus: all` 将 4090 直通进容器。

```bash
cd kronos_service
cp .env.example .env
# 编辑 .env：改 KRONOS_ADMIN_PASSWORD，建议生成 KRONOS_JWT_SECRET

docker compose up -d --build
docker compose logs -f          # 看到模型预加载完成即可用
```

- 服务地址：http://localhost:8000 （管理后台 http://localhost:8000/admin ，接口文档 http://localhost:8000/api/docs ）
- 挂载：`../models → /models`（权重）、`../volumes/instance → /app/instance`（SQLite）、`../volumes/logs → /app/logs`（日志），容器重启数据不丢
- 模型齐备后可在 `.env` 打开 `HF_HUB_OFFLINE=1`，启动不再联网探测
- 扩容：单卡跑多 worker 只会复制显存，需要更高吞吐请复制容器/挂多卡

## 3. 本地开发运行

```bash
pip install -r requirements.txt          # 仓库根（torch 按 CUDA 版本自装）
pip install -r kronos_service/requirements.txt
cd kronos_service
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

管理员账密从 `.env`（仓库根或 kronos_service/ 下）读取，模板见 `.env.example`。

## 4. 调用方式

### 签发密钥
管理后台 →「API 密钥」→ 签发（明文仅展示一次），请求头携带 `X-API-Key: kron-xxxx`。

### 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/v1/models` | 模型列表与加载状态 |
| POST | `/v1/predict` | 同步预测（等待结果直接返回） |
| POST | `/v1/predict/async` | 异步预测（202 + task_id） |
| GET | `/v1/tasks/{task_id}` | 任务查询（异步结果 / 同步超时兜底） |
| GET | `/healthz` `/readyz` | 探活（无需鉴权） |

交互式文档：http://localhost:8000/api/docs （OpenAPI，可直接试调）。

### 入参（POST /v1/predict 与 /v1/predict/async，JSON body）

| 字段 | 类型 | 必填 | 默认 | 约束 | 说明 |
|---|---|---|---|---|---|
| `data` | Candle[] | ✓ | — | 2~6000 行 | 历史K线数组；服务端按时间排序去重；**股票必须后复权**（见⚠️） |
| `pred_len` | int | ✓ | — | 1~480 | 预测步数（根数），另受系统配置 `max_pred_len` 限制 |
| `model` | string | — | 自动路由 | `kronos-base` / `kronos-as-base` | 显式指定则跳过自动路由（见「双模型自动路由」） |
| `asset_type` | string | — | — | `index` / `stock` | 标的类型显式声明，优先于 `symbol` 自动识别 |
| `symbol` | string | — | — | ≤32 字符 | 标的代码，用于路由与统计；指数请带前缀（`sh000300`） |
| `lookback` | int | — | 400 | 8~4096，且 ≤ 模型上下文 512 | 实际参与推理的历史窗口根数；不足时自动取全部 |
| `temperature` | float | — | 1.0 | (0, 5] | 采样温度，越低越保守 |
| `top_k` | int | — | 0 | 0~1024 | top-k 截断，0 = 不启用 |
| `top_p` | float | — | 0.9 | (0, 1] | 核采样截断 |
| `sample_count` | int | — | 1 | 1~8 | 采样路径数；**输出为多条路径的均值**（即回测 mean 信号） |
| `future_timestamps` | string[] | — | 服务端推断 | 长度 = pred_len | 显式未来时间戳（需要精确 A 股节假日日历时使用） |

`Candle` 结构：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `timestamp` | string | ✓ | ISO8601，如 `2026-11-14 14:35:00` |
| `open` `high` `low` `close` | float | ✓ | 四值必须 > 0；股票为**后复权价** |
| `volume` `amount` | float | — | 可省略，缺省补 0 |

### 出参

**同步 200**（`POST /v1/predict`）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `task_id` | string | 任务号；同步超时（408）后凭它查询结果 |
| `model` | string | 服务端**实际使用**的模型（路由结果） |
| `asset_type` | string\|null | 路由依据回显：`index` / `stock` / `null`（未识别或显式指定 model） |
| `symbol` | string\|null | 回显请求的 symbol |
| `request` | object | 实际生效参数：`{pred_len, lookback, temperature, top_k, top_p, sample_count}` |
| `input` | object | 输入摘要：`{from, to, points(实际参与根数), frequency}`；frequency ∈ `daily` / `intraday` / `generic` |
| `timings` | object | `{queue_ms, inference_ms, total_ms}` |
| `predictions` | Candle[] | `pred_len` 根预测K线（多路径均值，位于与入参相同的**后复权空间**） |

**异步 202**（`POST /v1/predict/async`）：`{task_id, status: "queued", status_url}`

**任务查询**（`GET /v1/tasks/{task_id}`）：`{task_id, status(queued/running/succeeded/failed), model, symbol, pred_len, queue_ms, inference_ms, created_at, finished_at, result: {predictions: [...]}, error}` —— `result` 仅成功时返回。

### 错误码

| 状态码 | 场景 | 处理建议 |
|---|---|---|
| 401 | 缺少 / 无效 `X-API-Key` | 核对请求头 |
| 403 | 密钥被禁用 / 吊销 / 过期 | 联系管理员 |
| 404 | 任务不存在或不属于该密钥 | 核对 task_id |
| 408 | 同步等待超时 | 用返回的 `task_id` 走 `/v1/tasks/{id}` 取结果，**不要重复提交** |
| 413 | 请求体超过 20MB | 压缩 `data` 行数 |
| 422 | 参数 / 数据校验失败 | 按 `detail` 修正（含 K 线含非正数、时间戳不可解析等） |
| 429 | 超密钥限速或当日配额 | 退避重试或提额 |
| 503 | 队列满 / 服务未就绪 / 模型权重未部署 | 稍后重试；权重缺失需先部署到 `models/` |

### 同步预测

```bash
curl -s http://localhost:8000/v1/predict \
  -H "X-API-Key: kron-你的密钥" -H "Content-Type: application/json" \
  -d '{
    "symbol": "300308",
    "data": [
      {"timestamp":"2026-11-14 14:35:00","open":11.2,"high":11.3,"low":11.1,"close":11.25,"volume":1000,"amount":11250},
      ...（≥32 根，建议 400 根左右，升序或乱序均可，服务端会排序；股票须后复权，见下方⚠️）
    ],
    "pred_len": 48,
    "temperature": 1.0, "top_p": 0.9, "sample_count": 1
  }'
```

响应（节选）——`model` 为服务端实际路由结果，`asset_type` 回显路由依据（`index`/`stock`/`null`）：

```json
{
  "task_id": "…", "model": "kronos-as-base", "asset_type": "stock", "symbol": "300308",
  "input": {"from": "…", "to": "…", "points": 400, "frequency": "intraday"},
  "timings": {"queue_ms": 3, "inference_ms": 1840, "total_ms": 1843},
  "predictions": [{"timestamp": "2026-11-16 09:35:00", "open": …, "high": …, "low": …, "close": …, "volume": …, "amount": …}]
}
```

指数预测（自动路由到原版 `kronos-base`）：

```bash
curl -s http://localhost:8000/v1/predict \
  -H "X-API-Key: kron-你的密钥" -H "Content-Type: application/json" \
  -d '{
    "symbol": "sh000300",           # 指数请带前缀, 见路由规则表
    "asset_type": "index",          # 可省; 裸 000xxx 代码歧义时必传
    "data": [ ...后复权/原始日线, 建议约 400 根... ],
    "pred_len": 10, "sample_count": 3
  }'
# → "model": "kronos-base", "asset_type": "index"
```

### 双模型自动路由

未显式传 `model` 时按标的类型分流（回测依据：指数微调在全部信号上劣于原版零样本，股票微调大幅改善股票预测——两模型各守所长）：

| 优先级 | 规则 | 结果 |
|---|---|---|
| 1 | 请求显式带 `model` | 直接使用该模型，跳过路由 |
| 2 | 请求带 `asset_type`（`index`/`stock`） | 按声明路由 |
| 3 | `symbol` 自动识别（下表） | 指数→`kronos-base`，股票→`kronos-as-base` |
| 4 | 无法识别 | 兜底 `default_model`（默认 `kronos-base`），`asset_type` 回显 `null` |

`symbol` 识别规则（大小写不敏感）：

| 写法 | 判定 | 示例 |
|---|---|---|
| `sh000xxx` / `sh999xxx` / `csi*` 前缀 | 指数 | `sh000300`、`sh999999`、`csih00922` |
| `sz399xxx` 前缀、`399xxx`（含后缀 `.SZ` 或裸码） | 指数 | `sz399001`、`399006` |
| `000xxx.SH` / `*.CSI` 后缀 | 指数 | `000905.SH`、`000922.CSI` |
| `60*` `68*` `30*` `002/003` 开头（裸码或带 `SH/SZ` 前后缀） | 股票 | `600519`、`300308`、`sz000001` |
| **裸 `000xxx`** | **歧义**（深市股票与指数共用段，如 `000001` 平安银行 vs 上证指数） | `000300` → 兜底默认模型；**指数请带前缀或传 `asset_type`** |

路由行为可在管理后台「系统配置」热调：`enable_asset_routing`（总开关）、`index_model`、`stock_model`。两个模型默认同时预加载（`preload_models`），合计显存约 0.9GB；路由到权重未部署且无 HF 回退的模型时返回 `503` 并提示部署。

**指数轮动打分**：参数配方与排序方法见下方「最佳使用方式 ③ 场景参数配方」。

### 异步预测

```bash
curl -s -X POST http://localhost:8000/v1/predict/async -H "X-API-Key: kron-…" -H "Content-Type: application/json" -d '{…同上…}'
# → {"task_id":"…","status":"queued","status_url":"/v1/tasks/…"}
curl -s http://localhost:8000/v1/tasks/<task_id> -H "X-API-Key: kron-…"
```

### 最佳使用方式

#### ① 复权规范（最重要）

⚠️ **股票入参 K 线必须后复权（hfq）**：A股微调模型（`kronos-as-base`）在后复权日线上训练——后复权把分红/送转的除权缺口直接折算进历史价格，保证价格序列连续、涨跌幅真实；传不复权或前复权数据会在除权日引入跳变，造成分布漂移、预测失真。预测输出同样位于后复权价格空间，取其**相对涨跌幅**使用即可，不要与行情原始价绝对值直接比对。指数无除权事件，原始价即等价于后复权，无需处理。

#### ② 模型选择

不传 `model` 让服务自动路由即可（指数→`kronos-base` 原版零样本，股票→`kronos-as-base` A股微调版，规则见上节）。仅在需要对照实验或指定小模型时显式传 `model`。裸 `000xxx` 代码（深市股票与指数共用段）务必带交易所前缀或显式 `asset_type`，否则走默认模型。

#### ③ 场景参数配方

| 场景 | lookback | pred_len | temperature | top_p | sample_count | 要点 |
|---|---|---|---|---|---|---|
| **指数轮动打分**（已回测验证） | 90~400 | 10 | 0.6 | 0.9 | 3 | 对 14 个指数各调一次，按「预测收盘均值相对末根收盘涨幅」降序取 TopN 持有轮动；即回测 mean 信号，含成本年化超额 +12.6%（vs 沪深300） |
| **个股日线走势** | 90~400 | 10 | 0.6~1.0 | 0.9 | 1~3 | AS-v1 微调分布为「90 根历史→10 根未来」，`pred_len=10` 最贴合训练分布；更远外推可用但超出验证范围 |
| **分钟线短线外推** | ~400 | ≤480 | 1.0 | 0.9 | 1 | 用原版 `kronos-base`（`model` 显式指定或 `asset_type` 缺省路由）；pred_len 大时走异步接口 |

#### ④ 数据准备

- 行数 ≥32 根（`min_lookback`），建议 400 根左右（`default_lookback`），上限 6000 行 / 请求体 20MB
- 升序或乱序均可（服务端会排序去重），但**不应有长期停牌等结构性空洞**
- 周期通用（日线/5min/15min…），模型按时间戳间隔自动感知频率；跨品种（股票/指数/期货/加密）均可
- 需要精确 A 股交易日历（节假日）时显式传 `future_timestamps`，否则服务端按「日线跳周末、分钟线复用日内节拍」推断

#### ⑤ 调用策略

- **同步**：日线/短线预测（pred_len≤120）秒级返回，优先用
- **异步**：长预测、批量标的（如 14 指数轮动一次性提交）；`202` 后轮询 `/v1/tasks/{id}`
- 同步收到 `408` 不要重发——原任务仍在跑，凭返回的 `task_id` 查询即可
- 批量调用注意密钥限速（429 时退避重试）；轮动类每日定时任务建议错峰

#### ⑥ 结果解读

- `predictions` 是多条采样路径的均值（`sample_count=3` 即回测验证的 mean 信号），已天然平滑
- 预测值在后复权空间：**用相对涨跌幅**（`预测收盘 / 末根收盘 - 1`），不要拿绝对价格与行情软件比
- `volume/amount` 为分布外推参考值，置信度低于价格
- 多标的横向比较（轮动排序）比单标的绝对点位预测可靠——这是生成式 K 线模型的优势用法

## 5. 管理后台

登录后五个视图：

| 视图 | 功能 |
|---|---|
| 仪表盘 | 调用量/成功率/P95 延迟/队列深度、24h 调用与延迟趋势图、模型状态、GPU 显存、CPU/内存、密钥与标的用量排行（10s 自动刷新） |
| 模型管理 | 加载/卸载全部注册模型（含A股微调版 kronos-as-base）、设默认模型、查看权重来源（本地目录 or HF Hub） |
| API 密钥 | 签发（限速/配额/有效期）、启停、编辑、吊销、单密钥调用明细 |
| 系统配置 | 业务参数热调（默认模型、路由开关与 index/stock_model、pred_len/sample_count 上限、同步超时、合批开关、日志保留等）；标注"需重启生效"的项重启后生效 |
| 服务自检 | 一键检测：GPU、模型目录完整性、模型冒烟预测、推理队列、数据库、服务目录 |

管理员账密配置在 `.env`（`KRONOS_ADMIN_USERNAME/PASSWORD`）；登录接口有防爆破限速（5 次失败锁 15 分钟）。

## 6. 常见问题

| 现象 | 处理 |
|---|---|
| 容器内 GPU 不可用 | Windows 需较新 NVIDIA 驱动 + Docker Desktop WSL2 后端；`docker compose exec kronos-api python -c "import torch;print(torch.cuda.is_available())"` 验证 |
| 首次启动很慢 | `./models` 为空时会从 HF Hub 下载（base 约 400MB），先跑第 1 节脚本预下载 |
| 管理后台样式/组件加载失败 | CDN 使用 npmmirror，若内网受限改 `static/admin/index.html` 为 jsdelivr/unpkg 或内网镜像 |
| 重启后管理台需重新登录 | `.env` 中固定 `KRONOS_JWT_SECRET` 即可保持登录态 |
| 同步接口偶发 408 | 队列拥塞或 pred_len 过大；改用异步接口，或在系统配置调大 `sync_timeout_seconds` |

## 7. 运维备忘

- 数据库：`volumes/instance/kronos.db`（备份即拷贝该文件；请求日志与已完成任务按配置自动清理）
- 日志：`volumes/logs/kronos_service.log`（10MB×5 轮转）+ `docker compose logs`
- 压测：`python kronos_service/scripts/load_test.py --url http://localhost:8000 --key kron-… --concurrency 20`
- 测试：`pytest tests/service/`（单测 mock 推理；`pytest -m integration` 需本地模型）
