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
- 模型权重挂载宿主机 `./models/`（本地优先，缺失走 HF Hub 下载，同样落在宿主机）
- SQLite(WAL) 存密钥/请求日志/任务/运行配置；日志双写控制台与 `logs/` 滚动文件

---

## 1. 准备模型权重

```bash
# 默认下载 kronos-base + 配套 Tokenizer（优先从本机 HF 缓存拷贝，缺失才联网）
python kronos_service/scripts/download_models.py

# 全部三档模型
python kronos_service/scripts/download_models.py --models kronos-mini,kronos-small,kronos-base
```

生成结构（`config.json` + `*.safetensors`，均为 `from_pretrained` 可直接读取的本地目录）：

```
models/
├── Kronos-base/  ├─ Kronos-small/  ├─ Kronos-mini/
├── Kronos-Tokenizer-base/  └─ Kronos-Tokenizer-2k/
```

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

### 同步预测

```bash
curl -s http://localhost:8000/v1/predict \
  -H "X-API-Key: kron-你的密钥" -H "Content-Type: application/json" \
  -d '{
    "symbol": "300308",
    "model": "kronos-base",
    "data": [
      {"timestamp":"2026-11-14 14:35:00","open":11.2,"high":11.3,"low":11.1,"close":11.25,"volume":1000,"amount":11250},
      ...（≥32 根，建议 400 根左右，升序或乱序均可，服务端会排序）
    ],
    "pred_len": 48,
    "temperature": 1.0, "top_p": 0.9, "sample_count": 1
  }'
```

响应（节选）：

```json
{
  "task_id": "…", "model": "kronos-base", "symbol": "300308",
  "input": {"from": "…", "to": "…", "points": 400, "frequency": "intraday"},
  "timings": {"queue_ms": 3, "inference_ms": 1840, "total_ms": 1843},
  "predictions": [{"timestamp": "2026-11-16 09:35:00", "open": …, "high": …, "low": …, "close": …, "volume": …, "amount": …}]
}
```

### 异步预测

```bash
curl -s -X POST http://localhost:8000/v1/predict/async -H "X-API-Key: kron-…" -H "Content-Type: application/json" -d '{…同上…}'
# → {"task_id":"…","status":"queued","status_url":"/v1/tasks/…"}
curl -s http://localhost:8000/v1/tasks/<task_id> -H "X-API-Key: kron-…"
```

### 说明与约束

- `data` 必含 `timestamp/open/high/low/close`，`volume/amount` 可省（补 0）；指数、股票、任意周期（日线/5min/15min…）通用，模型按日历特征感知频率
- 未来时间戳默认由服务端按历史节拍推断（日线跳周末、分钟线复用日内节拍）；需要精确交易日历（A股节假日）时显式传 `future_timestamps`（长度=pred_len）
- 同步等待超过 `sync_timeout_seconds` 返回 `408 + task_id`，结果稍后经 `/v1/tasks/{task_id}` 获取，不会丢
- 限流：密钥级每分钟限速 + 日配额，超限 `429`；队满 `503`；参数越界 `422`

## 5. 管理后台

登录后五个视图：

| 视图 | 功能 |
|---|---|
| 仪表盘 | 调用量/成功率/P95 延迟/队列深度、24h 调用与延迟趋势图、模型状态、GPU 显存、CPU/内存、密钥与标的用量排行（10s 自动刷新） |
| 模型管理 | 加载/卸载三档模型、设默认模型、查看权重来源（本地目录 or HF Hub） |
| API 密钥 | 签发（限速/配额/有效期）、启停、编辑、吊销、单密钥调用明细 |
| 系统配置 | 业务参数热调（默认模型、pred_len/sample_count 上限、同步超时、合批开关、日志保留等）；标注"需重启生效"的项重启后生效 |
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
