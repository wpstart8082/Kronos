/* Kronos 预测服务管理后台 SPA (Vue3 + Element Plus + ECharts, CDN 全局构建, 零构建工具链) */
/* global Vue, ElementPlus, ElementPlusIconsVue, echarts */

const { createApp, ref, reactive, computed, onMounted, onBeforeUnmount, watch, nextTick } = Vue;

// ---------------------------------------------------------------- API helper
async function api(method, url, body) {
  const headers = { "Content-Type": "application/json" };
  const token = localStorage.getItem("kronos_admin_token");
  if (token) headers["Authorization"] = "Bearer " + token;
  const resp = await fetch(url, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (resp.status === 401 && !url.endsWith("/login")) {
    localStorage.removeItem("kronos_admin_token");
    throw Object.assign(new Error("登录已过期，请重新登录"), { fatal401: true });
  }
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const detail = data.detail;
    const msg =
      typeof detail === "string"
        ? detail
        : detail && detail.message
          ? detail.message
          : JSON.stringify(detail || {});
    throw Object.assign(new Error(msg || `HTTP ${resp.status}`), { status: resp.status, data });
  }
  return data;
}

function fmtTs(iso) {
  if (!iso) return "-";
  try {
    const d = new Date(iso);
    const p = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
  } catch {
    return iso;
  }
}
function fmtUptime(s) {
  if (s == null) return "-";
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h > 0 ? `${h}小时${m}分` : `${m}分${s % 60}秒`;
}

// ---------------------------------------------------------------- Login
const LoginView = {
  emits: ["logged-in"],
  setup(props, { emit }) {
    const form = reactive({ username: "", password: "" });
    const loading = ref(false);
    const formRef = ref(null);
    const doLogin = async () => {
      loading.value = true;
      try {
        const r = await api("POST", "/api/admin/login", { ...form });
        localStorage.setItem("kronos_admin_token", r.token);
        emit("logged-in", r);
      } catch (e) {
        ElementPlus.ElMessage.error(e.message);
      } finally {
        loading.value = false;
      }
    };
    return { form, loading, doLogin, formRef };
  },
  template: `
  <div class="login-wrap">
    <div class="login-card">
      <div class="login-title">📈 Kronos 预测服务</div>
      <div class="login-sub">管理控制台 · Kronos Prediction Service Console</div>
      <el-form :model="form" @submit.prevent="doLogin">
        <el-form-item>
          <el-input v-model="form.username" placeholder="管理员账号" size="large" autofocus>
            <template #prefix><el-icon><User /></el-icon></template>
          </el-input>
        </el-form-item>
        <el-form-item>
          <el-input v-model="form.password" type="password" placeholder="密码" size="large" show-password @keyup.enter="doLogin">
            <template #prefix><el-icon><Lock /></el-icon></template>
          </el-input>
        </el-form-item>
        <el-button type="primary" size="large" style="width:100%" :loading="loading" @click="doLogin">登 录</el-button>
      </el-form>
    </div>
  </div>`,
};

// ---------------------------------------------------------------- Dashboard
const Dashboard = {
  setup() {
    const ov = ref(null);
    const loading = ref(false);
    const callsChartEl = ref(null);
    let callsChart = null, timer = null;

    const refresh = async () => {
      try {
        ov.value = await api("GET", "/api/admin/overview");
      } catch (e) {
        ElementPlus.ElMessage.error(e.message);
      }
    };

    const renderCharts = () => {
      if (!ov.value || !callsChartEl.value) return;
      if (!callsChart) callsChart = echarts.init(callsChartEl.value);
      const hours = ov.value.hourly_series.map((x) => x.hour.slice(5, 13).replace("T", " "));
      callsChart.setOption({
        tooltip: { trigger: "axis" },
        legend: { data: ["成功", "失败", "平均耗时(ms)"], top: 0 },
        grid: { left: 50, right: 55, top: 36, bottom: 30 },
        xAxis: { type: "category", data: hours, axisLabel: { fontSize: 10 } },
        yAxis: [
          { type: "value", name: "调用次数", minInterval: 1 },
          { type: "value", name: "ms", splitLine: { show: false } },
        ],
        series: [
          { name: "成功", type: "bar", stack: "calls", barMaxWidth: 26, itemStyle: { color: "#409eff" }, data: ov.value.hourly_series.map((x) => x.ok) },
          { name: "失败", type: "bar", stack: "calls", barMaxWidth: 26, itemStyle: { color: "#f56c6c" }, data: ov.value.hourly_series.map((x) => x.error) },
          { name: "平均耗时(ms)", type: "line", yAxisIndex: 1, smooth: true, itemStyle: { color: "#e6a23c" }, data: ov.value.hourly_series.map((x) => x.latency_avg_ms) },
        ],
      });
    };

    onMounted(async () => {
      await refresh();
      await nextTick();
      renderCharts();
      timer = setInterval(async () => {
        await refresh();
        renderCharts();
      }, 10000);
      window.addEventListener("resize", onResize);
    });
    onBeforeUnmount(() => {
      if (timer) clearInterval(timer);
      window.removeEventListener("resize", onResize);
      if (callsChart) callsChart.dispose();
    });
    const onResize = () => callsChart && callsChart.resize();

    const gpuPct = computed(() => {
      const g = ov.value?.gpu;
      if (!g || !g.available || !g.total_mem_gb) return 0;
      return Math.min(100, Math.round((g.used_mem_gb / g.total_mem_gb) * 100));
    });

    return { ov, loading, callsChartEl, fmtUptime, gpuPct };
  },
  template: `
  <div v-if="ov">
    <div class="page-title">仪表盘</div>

    <el-row :gutter="14" class="stat-cards">
      <el-col :span="6"><el-card shadow="never" class="stat-card">
        <div class="num">{{ ov.requests_24h.total }}</div><div class="label">24小时调用</div>
        <div class="muted">近1小时 {{ ov.requests_1h.total }} 次</div>
      </el-card></el-col>
      <el-col :span="6"><el-card shadow="never" class="stat-card">
        <div class="num" :class="ov.requests_24h.success_rate !== null && ov.requests_24h.success_rate < 95 ? 'warn' : ''">
          {{ ov.requests_24h.success_rate === null ? "-" : ov.requests_24h.success_rate + "%" }}</div>
        <div class="label">24小时成功率</div>
        <div class="muted">4xx {{ ov.requests_24h.client_errors }} / 5xx {{ ov.requests_24h.server_errors }}</div>
      </el-card></el-col>
      <el-col :span="6"><el-card shadow="never" class="stat-card">
        <div class="num">{{ ov.requests_24h.latency_p95_ms }} ms</div><div class="label">延迟 P95 (24h)</div>
        <div class="muted">P50 {{ ov.requests_24h.latency_p50_ms }}ms · 均值 {{ ov.requests_24h.latency_avg_ms }}ms</div>
      </el-card></el-col>
      <el-col :span="6"><el-card shadow="never" class="stat-card">
        <div class="num" :class="ov.queue.queued > 10 ? 'warn' : ''">{{ ov.queue.queued }}</div><div class="label">排队中 / 执行中 {{ ov.queue.active }}</div>
        <div class="muted">等待P95 {{ ov.queue.wait_ms_p95 }}ms · 推理P95 {{ ov.queue.infer_ms_p95 }}ms</div>
      </el-card></el-col>
    </el-row>

    <el-card shadow="never" class="chart-card">
      <template #header><b>调用量与延迟趋势（近24小时）</b></template>
      <div ref="callsChartEl" class="chart"></div>
    </el-card>

    <el-row :gutter="14">
      <el-col :span="8">
        <el-card shadow="never" class="chart-card">
          <template #header><b>模型状态</b></template>
          <el-table :data="ov.models" size="small">
            <el-table-column prop="key" label="模型" width="110"></el-table-column>
            <el-table-column label="状态" width="80">
              <template #default="{row}">
                <el-tag v-if="row.loaded" type="success" size="small">已加载</el-tag>
                <el-tag v-else-if="row.loading" type="warning" size="small">加载中</el-tag>
                <el-tag v-else type="info" size="small">未加载</el-tag>
              </template>
            </el-table-column>
            <el-table-column label="权重来源" width="90">
              <template #default="{row}">
                <el-tag v-if="row.location==='local'" size="small" effect="plain">本地目录</el-tag>
                <el-tag v-else size="small" effect="plain" type="warning">HF Hub</el-tag>
              </template>
            </el-table-column>
            <el-table-column label="执行中/累计" min-width="100">
              <template #default="{row}">{{ row.loaded ? (row.active_inferences + ' / ' + row.total_inferences) : '-' }}</template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card shadow="never" class="chart-card">
          <template #header><b>GPU / 系统资源</b></template>
          <template v-if="ov.gpu.available">
            <div><b>{{ ov.gpu.device_name }}</b></div>
            <el-progress class="gpu-bar" :percentage="gpuPct" :stroke-width="14"
              :format="() => ov.gpu.used_mem_gb + ' / ' + ov.gpu.total_mem_gb + ' GB'"></el-progress>
          </template>
          <el-alert v-else type="warning" :closable="false" title="GPU 不可用（CPU 回退模式）" show-icon></el-alert>
          <el-divider style="margin:12px 0"></el-divider>
          <div class="muted">CPU 使用率：{{ ov.system.cpu_percent }}%　内存：{{ ov.system.mem_used_gb }}/{{ ov.system.mem_total_gb }} GB（{{ ov.system.mem_percent }}%）</div>
          <div class="muted mt">运行时长：{{ fmtUptime(ov.service.uptime_s) }}　版本：v{{ ov.service.version }}</div>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card shadow="never" class="chart-card">
          <template #header><b>用量排行（24h）</b></template>
          <div class="muted">按密钥</div>
          <div v-for="t in ov.top_keys" :key="'k'+t.name" class="muted mt" style="color:#303133">
            <el-text size="small">{{ t.name }}</el-text>
            <el-tag size="small" style="float:right">{{ t.count }}</el-tag>
          </div>
          <el-divider style="margin:10px 0"></el-divider>
          <div class="muted">按标的 symbol</div>
          <div v-for="t in ov.top_symbols" :key="'s'+t.name" class="muted mt" style="color:#303133">
            <el-text size="small">{{ t.name }}</el-text>
            <el-tag size="small" type="info" style="float:right">{{ t.count }}</el-tag>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>`,
};

// ---------------------------------------------------------------- API Keys
const KeysView = {
  setup() {
    const keys = ref([]);
    const loading = ref(false);
    const createVisible = ref(false);
    const created = ref(null); // { key, plaintext }
    const usageVisible = ref(false);
    const usage = ref(null);
    const editVisible = ref(false);
    const editForm = reactive({ id: 0, name: "", note: "", enabled: true, rate_limit_per_min: 0, daily_quota: 0 });
    const createForm = reactive({ name: "", note: "", rate_limit_per_min: 60, daily_quota: 0, expires_days: null });

    const refresh = async () => {
      loading.value = true;
      try {
        const r = await api("GET", "/api/admin/keys");
        keys.value = r.keys;
      } catch (e) {
        ElementPlus.ElMessage.error(e.message);
      } finally {
        loading.value = false;
      }
    };
    onMounted(refresh);

    const doCreate = async () => {
      try {
        created.value = await api("POST", "/api/admin/keys", { ...createForm, expires_days: createForm.expires_days || null });
        createVisible.value = false;
        ElementPlus.ElMessage.success("密钥已创建，请立即复制保存（仅展示一次）");
        Object.assign(createForm, { name: "", note: "", rate_limit_per_min: 60, daily_quota: 0, expires_days: null });
        refresh();
      } catch (e) {
        ElementPlus.ElMessage.error(e.message);
      }
    };
    const copyKey = async () => {
      try {
        await navigator.clipboard.writeText(created.value.plaintext);
        ElementPlus.ElMessage.success("已复制到剪贴板");
      } catch {
        ElementPlus.ElMessage.warning("复制失败，请手动选择复制");
      }
    };
    const openEdit = (row) => {
      Object.assign(editForm, {
        id: row.id, name: row.name, note: row.note, enabled: row.enabled,
        rate_limit_per_min: row.rate_limit_per_min, daily_quota: row.daily_quota,
      });
      editVisible.value = true;
    };
    const doEdit = async () => {
      try {
        await api("PATCH", "/api/admin/keys/" + editForm.id, {
          name: editForm.name, note: editForm.note, enabled: editForm.enabled,
          rate_limit_per_min: editForm.rate_limit_per_min, daily_quota: editForm.daily_quota,
        });
        editVisible.value = false;
        ElementPlus.ElMessage.success("已保存");
        refresh();
      } catch (e) {
        ElementPlus.ElMessage.error(e.message);
      }
    };
    const revoke = async (row) => {
      try {
        await ElementPlus.ElMessageBox.confirm(`确定吊销密钥「${row.name}」？吊销后立即失效且不可恢复`, "吊销确认", { type: "warning" });
        await api("DELETE", "/api/admin/keys/" + row.id);
        ElementPlus.ElMessage.success("已吊销");
        refresh();
      } catch (e) {
        if (e !== "cancel" && e.message) ElementPlus.ElMessage.error(e.message);
      }
    };
    const toggleEnabled = async (row) => {
      try {
        await api("PATCH", "/api/admin/keys/" + row.id, { enabled: row.enabled });
        refresh();
      } catch (e) {
        ElementPlus.ElMessage.error(e.message);
        refresh();
      }
    };
    const showUsage = async (row) => {
      usage.value = await api("GET", `/api/admin/keys/${row.id}/usage?limit=100`);
      usageVisible.value = true;
    };

    return {
      keys, loading, createVisible, createForm, created, doCreate, copyKey,
      editVisible, editForm, openEdit, doEdit, revoke, toggleEnabled,
      usageVisible, usage, showUsage, fmtTs,
    };
  },
  template: `
  <div>
    <div class="page-title">API 密钥管理
      <el-button type="primary" size="small" style="float:right" @click="createVisible = true">+ 签发新密钥</el-button>
    </div>
    <el-card shadow="never">
      <el-table :data="keys" v-loading="loading" size="default">
        <el-table-column prop="name" label="名称" min-width="130"></el-table-column>
        <el-table-column prop="key_prefix" label="密钥前缀" width="130">
          <template #default="{row}"><span class="mono">{{ row.key_prefix }}…</span></template>
        </el-table-column>
        <el-table-column label="状态" width="80">
          <template #default="{row}">
            <el-switch :model-value="row.enabled" @change="row.enabled = !row.enabled; toggleEnabled(row)" size="small"></el-switch>
          </template>
        </el-table-column>
        <el-table-column label="限速/分钟" width="95">
          <template #default="{row}">{{ row.rate_limit_per_min || '不限' }}</template>
        </el-table-column>
        <el-table-column label="日配额" width="90">
          <template #default="{row}">{{ row.daily_quota || '不限' }}</template>
        </el-table-column>
        <el-table-column label="过期时间" width="150">
          <template #default="{row}">{{ row.expires_at ? fmtTs(row.expires_at) : '永久' }}</template>
        </el-table-column>
        <el-table-column label="今日/累计调用" width="110">
          <template #default="{row}">{{ row.total_calls }}</template>
        </el-table-column>
        <el-table-column label="最近使用" width="150">
          <template #default="{row}">{{ fmtTs(row.last_used_at) }}</template>
        </el-table-column>
        <el-table-column label="操作" width="200" fixed="right">
          <template #default="{row}">
            <el-button size="small" text type="primary" @click="showUsage(row)">用量</el-button>
            <el-button size="small" text @click="openEdit(row)">编辑</el-button>
            <el-button size="small" text type="danger" @click="revoke(row)">吊销</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog v-model="createVisible" title="签发新密钥" width="480px">
      <el-form :model="createForm" label-width="100px">
        <el-form-item label="名称" required><el-input v-model="createForm.name" placeholder="如: 量化平台A"></el-input></el-form-item>
        <el-form-item label="备注"><el-input v-model="createForm.note" type="textarea" :rows="2"></el-input></el-form-item>
        <el-form-item label="限速(次/分)"><el-input-number v-model="createForm.rate_limit_per_min" :min="0" :max="10000"></el-input-number>
          <div class="muted">0 = 不限制</div></el-form-item>
        <el-form-item label="日配额(次)"><el-input-number v-model="createForm.daily_quota" :min="0" :max="10000000"></el-input-number>
          <div class="muted">0 = 不限制</div></el-form-item>
        <el-form-item label="有效天数"><el-input-number v-model="createForm.expires_days" :min="1" :max="3650" placeholder="留空=永久"></el-input-number>
          <div class="muted">留空 = 永久有效</div></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="createVisible = false">取消</el-button>
        <el-button type="primary" @click="doCreate">签发</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="editVisible" title="编辑密钥" width="460px">
      <el-form :model="editForm" label-width="100px">
        <el-form-item label="名称"><el-input v-model="editForm.name"></el-input></el-form-item>
        <el-form-item label="备注"><el-input v-model="editForm.note" type="textarea" :rows="2"></el-input></el-form-item>
        <el-form-item label="启用"><el-switch v-model="editForm.enabled"></el-switch></el-form-item>
        <el-form-item label="限速(次/分)"><el-input-number v-model="editForm.rate_limit_per_min" :min="0" :max="10000"></el-input-number></el-form-item>
        <el-form-item label="日配额(次)"><el-input-number v-model="editForm.daily_quota" :min="0" :max="10000000"></el-input-number></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="editVisible = false">取消</el-button>
        <el-button type="primary" @click="doEdit">保存</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="usageVisible" :title="'用量明细 - ' + (usage?.key?.name || '')" width="860px">
      <el-descriptions v-if="usage" :column="3" size="small" border>
        <el-descriptions-item label="今日调用">{{ usage.today_calls }}</el-descriptions-item>
        <el-descriptions-item label="累计调用">{{ usage.key.total_calls }}</el-descriptions-item>
        <el-descriptions-item label="限速">{{ usage.key.rate_limit_per_min || '不限' }}/分钟</el-descriptions-item>
      </el-descriptions>
      <el-table v-if="usage" :data="usage.recent_logs" size="small" class="mt" max-height="420">
        <el-table-column label="时间" width="150"><template #default="{row}">{{ fmtTs(row.ts) }}</template></el-table-column>
        <el-table-column prop="endpoint" label="接口" width="130"></el-table-column>
        <el-table-column label="状态" width="70">
          <template #default="{row}">
            <el-tag :type="row.status_code < 400 ? 'success' : 'danger'" size="small">{{ row.status_code }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="latency_ms" label="耗时ms" width="80"></el-table-column>
        <el-table-column prop="model" label="模型" width="100"></el-table-column>
        <el-table-column prop="symbol" label="标的" width="90"></el-table-column>
        <el-table-column prop="pred_len" label="步数" width="60"></el-table-column>
        <el-table-column prop="error" label="错误" min-width="160" show-overflow-tooltip></el-table-column>
      </el-table>
    </el-dialog>

    <el-dialog v-if="created" :model-value="true" title="新密钥已生成" width="520px" :close-on-click-modal="false" @update:model-value="created = null">
      <el-alert type="warning" :closable="false" title="明文密钥仅此一次展示，请立即复制保存！" show-icon></el-alert>
      <div class="key-plaintext mono">{{ created.plaintext }}</div>
      <div class="muted">调用方式：请求头携带 <code>X-API-Key: {{ created.plaintext }}</code></div>
      <template #footer>
        <el-button @click="copyKey">复制密钥</el-button>
        <el-button type="primary" @click="created = null">我已保存</el-button>
      </template>
    </el-dialog>
  </div>`,
};

// ---------------------------------------------------------------- Config
const ConfigView = {
  setup() {
    const items = ref([]);
    const saving = ref(false);
    const refresh = async () => {
      try {
        const r = await api("GET", "/api/admin/config");
        items.value = r.config.map((c) => ({ ...c, draft: Array.isArray(c.value) ? c.value.join(",") : c.value }));
      } catch (e) {
        ElementPlus.ElMessage.error(e.message);
      }
    };
    onMounted(refresh);
    const save = async () => {
      saving.value = true;
      try {
        const payload = {};
        for (const it of items.value) payload[it.key] = it.type === "list" ? String(it.draft) : it.draft;
        const r = await api("PUT", "/api/admin/config", payload);
        ElementPlus.ElMessage.success(
          r.restart_required.length ? `已保存；${r.restart_required.join("、")} 需重启服务后生效` : "配置已保存并生效",
        );
        refresh();
      } catch (e) {
        ElementPlus.ElMessage.error(e.message);
      } finally {
        saving.value = false;
      }
    };
    return { items, save, saving };
  },
  template: `
  <div>
    <div class="page-title">系统配置
      <el-button type="primary" size="small" style="float:right" :loading="saving" @click="save">保存全部</el-button>
    </div>
    <el-card shadow="never">
      <el-form label-width="200px">
        <el-form-item v-for="it in items" :key="it.key" :label="it.key">
          <div style="width:100%">
            <el-switch v-if="it.type === 'bool'" v-model="it.draft"></el-switch>
            <el-input-number v-else-if="it.type === 'int'" v-model="it.draft" style="width:200px"></el-input-number>
            <el-input v-else v-model="it.draft" style="width:320px" :placeholder="it.type==='list' ? '逗号分隔' : ''"></el-input>
            <el-tag v-if="it.restart_required" size="small" type="warning" style="margin-left:8px">需重启生效</el-tag>
            <div class="muted">{{ it.desc }}</div>
          </div>
        </el-form-item>
      </el-form>
    </el-card>
  </div>`,
};

// ---------------------------------------------------------------- Selftest
const SelftestView = {
  setup() {
    const running = ref(false);
    const result = ref(null);
    const run = async () => {
      running.value = true;
      result.value = null;
      try {
        result.value = await api("POST", "/api/admin/selftest");
      } catch (e) {
        ElementPlus.ElMessage.error(e.message);
      } finally {
        running.value = false;
      }
    };
    return { running, result, run, fmtTs };
  },
  template: `
  <div>
    <div class="page-title">服务自检
      <el-button type="primary" size="small" style="float:right" :loading="running" @click="run">
        {{ running ? '检测中...（冒烟预测可能需要数十秒）' : '▶ 立即执行自检' }}
      </el-button>
    </div>
    <el-card shadow="never">
      <template v-if="!result && !running">
        <el-empty description="点击右上角按钮执行自检：GPU 检测 / 模型目录 / 模型冒烟预测 / 推理队列 / 数据库 / 服务目录"></el-empty>
      </template>
      <template v-else-if="running">
        <el-skeleton :rows="6" animated></el-skeleton>
      </template>
      <template v-else>
        <el-result :icon="result.passed ? 'success' : 'error'" :title="result.summary" :sub-title="'总耗时 ' + result.total_ms + 'ms · ' + fmtTs(result.ran_at)"></el-result>
        <div v-for="c in result.checks" :key="c.name" class="selftest-item">
          <el-tag :type="c.passed ? 'success' : 'danger'" size="small">{{ c.passed ? '通过' : '失败' }}</el-tag>
          <b style="margin-left:8px">{{ c.name }}</b>
          <span class="muted" style="float:right">{{ c.latency_ms }}ms</span>
          <div class="selftest-detail">{{ c.detail }}</div>
        </div>
      </template>
    </el-card>
  </div>`,
};

// ---------------------------------------------------------------- Models
const ModelsView = {
  setup() {
    const data = ref(null);
    const loading = ref(false);
    const refresh = async () => {
      try {
        data.value = await api("GET", "/api/admin/models");
      } catch (e) {
        ElementPlus.ElMessage.error(e.message);
      }
    };
    onMounted(refresh);
    let timer = setInterval(refresh, 5000);
    onBeforeUnmount(() => clearInterval(timer));
    const doLoad = async (key) => {
      try {
        const r = await api("POST", `/api/admin/models/${key}/load`);
        ElementPlus.ElMessage.info(r.hint);
        refresh();
      } catch (e) {
        ElementPlus.ElMessage.error(e.message);
      }
    };
    const doUnload = async (key) => {
      try {
        await api("POST", `/api/admin/models/${key}/unload`);
        ElementPlus.ElMessage.success("卸载指令已下发");
        refresh();
      } catch (e) {
        ElementPlus.ElMessage.error(e.message);
      }
    };
    const setDefault = async (key) => {
      try {
        await api("PUT", "/api/admin/config", { default_model: key });
        ElementPlus.ElMessage.success("默认模型已更新");
        refresh();
      } catch (e) {
        ElementPlus.ElMessage.error(e.message);
      }
    };
    return { data, doLoad, doUnload, setDefault, loading };
  },
  template: `
  <div>
    <div class="page-title">模型管理</div>
    <el-row :gutter="14" v-if="data">
      <el-col :span="8" v-for="m in data.models" :key="m.key">
        <el-card shadow="never" class="chart-card">
          <template #header>
            <b>{{ m.display_name }}</b>
            <el-tag v-if="m.key === data.default_model" size="small" style="margin-left:6px">默认</el-tag>
            <el-tag v-if="m.loaded" type="success" size="small" style="float:right">已加载</el-tag>
            <el-tag v-else-if="m.loading" type="warning" size="small" style="float:right">加载中</el-tag>
            <el-tag v-else type="info" size="small" style="float:right">未加载</el-tag>
          </template>
          <el-descriptions :column="1" size="small">
            <el-descriptions-item label="模型标识">{{ m.key }}</el-descriptions-item>
            <el-descriptions-item label="参数量">{{ m.params }}</el-descriptions-item>
            <el-descriptions-item label="上下文长度">{{ m.max_context }}</el-descriptions-item>
            <el-descriptions-item label="权重来源">
              <el-tag v-if="m.location==='local'" size="small" effect="plain">宿主机本地目录</el-tag>
              <el-tag v-else size="small" effect="plain" type="warning">HF Hub（联网下载）</el-tag>
            </el-descriptions-item>
            <el-descriptions-item v-if="m.loaded" label="设备">{{ m.device }} · 执行中 {{ m.active_inferences }} · 累计 {{ m.total_inferences }}</el-descriptions-item>
          </el-descriptions>
          <div style="margin-top:10px">
            <el-button size="small" type="primary" :disabled="m.loaded || m.loading" @click="doLoad(m.key)">加载</el-button>
            <el-button size="small" :disabled="!m.loaded" @click="doUnload(m.key)">卸载</el-button>
            <el-button size="small" text type="primary" :disabled="m.key === data.default_model" @click="setDefault(m.key)">设为默认</el-button>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>`,
};

// ---------------------------------------------------------------- App root
const App = {
  components: { LoginView, Dashboard, KeysView, ConfigView, SelftestView, ModelsView },
  setup() {
    const token = ref(localStorage.getItem("kronos_admin_token") || "");
    const view = ref("dashboard");
    const menu = [
      { key: "dashboard", label: "仪表盘", icon: "Odometer" },
      { key: "models", label: "模型管理", icon: "Cpu" },
      { key: "keys", label: "API 密钥", icon: "Key" },
      { key: "config", label: "系统配置", icon: "Setting" },
      { key: "selftest", label: "服务自检", icon: "CircleCheck" },
    ];
    const onLoggedIn = () => {
      token.value = localStorage.getItem("kronos_admin_token");
    };
    const logout = () => {
      localStorage.removeItem("kronos_admin_token");
      token.value = "";
    };
    return { token, view, menu, onLoggedIn, logout };
  },
  template: `
  <div style="height:100%">
    <LoginView v-if="!token" @logged-in="onLoggedIn"></LoginView>
    <el-container v-else class="layout">
      <el-aside width="216px" class="aside">
        <div class="brand">📈 Kronos 控制台<small>预测服务管理后台</small></div>
        <el-menu :default-active="view" background-color="#1f2d3d" text-color="#cfd8e3" active-text-color="#ffffff" @select="(k) => view = k">
          <el-menu-item v-for="m in menu" :key="m.key" :index="m.key">
            <el-icon><component :is="m.icon"></component></el-icon>
            <span>{{ m.label }}</span>
          </el-menu-item>
        </el-menu>
        <div class="aside-footer">
          <el-button text size="small" style="color:#7f95ab" @click="logout">退出登录</el-button>
        </div>
      </el-aside>
      <el-main class="main">
        <Dashboard v-if="view==='dashboard'"></Dashboard>
        <ModelsView v-else-if="view==='models'"></ModelsView>
        <KeysView v-else-if="view==='keys'"></KeysView>
        <ConfigView v-else-if="view==='config'"></ConfigView>
        <SelftestView v-else-if="view==='selftest'"></SelftestView>
      </el-main>
    </el-container>
  </div>`,
};

// ---------------------------------------------------------------- bootstrap
const app = createApp(App);
app.use(ElementPlus);
for (const [name, comp] of Object.entries(ElementPlusIconsVue)) app.component(name, comp);
app.mount("#app");
