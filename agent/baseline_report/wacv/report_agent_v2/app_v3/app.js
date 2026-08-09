"use strict";

const state = {
  view: "dashboard",
  dashboard: null,
  alerts: [],
  reports: null,
  genLogs: null,
  spatial: null,
  knowledge: null,
  audit: null,
  retrospective: null,
  filters: { level: "", country: "", crop: "", status: "", q: "", limit: 50 },
  detail: null,
  detailTab: "evidence",
  reportTab: "farmer",
  provider: "local",
  reportFilters: { generator: "", status: "", country: "", crop: "", q: "" },
};

const $ = (id) => document.getElementById(id);

const COUNTRY_ZH = { Argentina: "阿根廷", Brazil: "巴西", Germany: "德国", Uruguay: "乌拉圭" };
const CROP_ZH = { corn: "玉米", soybean: "大豆", wheat: "小麦", rapeseed: "油菜" };
const LEVEL_META = {
  action: { label: "行动告警", cls: "action" },
  watch: { label: "关注复查", cls: "watch" },
  normal: { label: "常规监测", cls: "normal" },
};
const WORKFLOW_ZH = {
  new: "新告警", assigned: "待现场核验", in_field: "巡田中",
  feedback_received: "已反馈", under_review: "农艺审核", closed: "已关闭",
};
const REPORT_STATUS_ZH = {
  needs_review: "待审核", approved: "已批准", rejected: "已驳回",
  needs_revision: "待修改", not_generated: "未生成",
};
const GENERATOR_ZH = { deterministic_fallback: "规则生成", llm_guarded: "智能润色" };
const PROVIDER_ZH = { local: "规则引擎", deepseek: "深度求索", openai: "OpenAI", anthropic: "Anthropic" };
const AUDIT_STATUS_ZH = { FAIL_FINAL: "未通过（如实保留）", PASS_FINAL: "通过" };
const AUDIT_ACTION_ZH = {
  seed_risk_events: "初始化风险事件", generate_risk_reports: "批量生成报告", generate_reports: "批量生成报告",
  generate_report: "生成报告", review_report: "审核报告", create_observation_snapshot: "保存现场快照",
  add_field_observation: "提交现场记录", assign_event: "分派任务", transition_event: "推进状态",
  complete_task: "完成任务", seed_commercial_workflow: "初始化演示流程",
};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function countryZh(country) { return COUNTRY_ZH[country] || country; }
function cropZh(crop) { return CROP_ZH[crop] || crop; }
function displayName(identity) {
  return `${countryZh(identity.country)} · ${cropZh(identity.crop)} · ${identity.season} 季 · 田块 ${identity.field_number}`;
}

function toast(message, error = false) {
  const node = $("toast");
  node.textContent = message;
  node.className = `toast show${error ? " error" : ""}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.className = "toast"; }, 3500);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `请求失败：${response.status}`);
  return payload;
}

function badge(level, label) {
  const meta = LEVEL_META[level] || { label: level || "—", cls: "normal" };
  return el("span", `badge ${meta.cls}`, label || meta.label);
}

function setTitle(title, subtitle) {
  $("pageTitle").textContent = title;
  $("pageSubtitle").textContent = subtitle || "";
}

/* ================= 总览 ================= */
function renderDashboard() {
  setTitle("总览", "322 个真实田块的低产风险预警组合与权威审计结果");
  const root = $("viewRoot");
  root.replaceChildren();
  const d = state.dashboard;
  if (!d) { root.append(el("div", "empty", "数据加载失败，请点击右上角刷新")); return; }
  const p = d.portfolio;
  const a = d.authoritative;
  const kpis = [
    ["监测田块", p.total_fields, "覆盖四国四作物", "normal"],
    ["行动告警", p.action_alerts, "连续两期越界，需农艺复核", "action"],
    ["关注复查", p.watch_alerts, "单期越界，建议优先巡田", "watch"],
    ["待审报告", p.pending_review_reports, "已生成，等待人工批准", "normal"],
  ];
  const grid = el("div", "grid grid-4");
  kpis.forEach(([label, value, note, cls]) => {
    const card = el("div", `kpi ${cls}`);
    card.append(el("span", "kpi-label", label), el("strong", "kpi-value", String(value)), el("span", "kpi-note", note));
    grid.append(card);
  });
  root.append(grid);

  const stats = el("div", "grid grid-3");
  stats.append(distributionCard("告警分布", p.risk_counts, [
    ["action", "行动告警"], ["watch", "关注复查"], ["normal", "常规监测"],
  ], { action: "var(--action)", watch: "var(--watch)", normal: "var(--brand)" }));
  stats.append(distributionCard("国家分布", p.country_counts, null, null, countryZh));
  stats.append(distributionCard("作物分布", p.crop_counts, null, null, cropZh));
  root.append(stats);

  const audit = el("div", "card");
  audit.append(el("h3", "card-title", "权威审计结果", el("span", "card-sub", "一次性冻结，不可改写")));
  const auditLines = [
    `审计状态：${AUDIT_STATUS_ZH[a.audit_status] || a.audit_status}（${a.audit_status === "FAIL_FINAL" ? "仅最差分组表现未达标，如实保留" : "全部门禁通过"}）`,
    `审计田块：${a.audit_fields} 个；第三观测期：均方根误差 ${fmt(a.look3_rmse, 3)} 吨/公顷，决定系数 ${fmt(a.look3_r2, 4)}，排名能力 ${fmt(a.look3_auroc, 4)}`,
    `修正口径·关注：误报率 ${(a.clean_null_watch.fpr * 100).toFixed(2)}%，敏感度 ${(a.clean_null_watch.sensitivity * 100).toFixed(2)}%，精确率 ${(a.clean_null_watch.precision * 100).toFixed(2)}%`,
    `修正口径·行动：误报率 ${(a.clean_null_action.fpr * 100).toFixed(2)}%，敏感度 ${(a.clean_null_action.sensitivity * 100).toFixed(2)}%，精确率 ${(a.clean_null_action.precision * 100).toFixed(2)}%`,
  ];
  auditLines.forEach((line) => audit.append(el("p", "", line)));
  audit.append(el("p", "notice", a.disclosure));
  root.append(audit);

  const lower = el("div", "grid grid-2");
  const workflow = el("div", "card");
  workflow.append(el("h3", "card-title", "工作流状态"));
  const counts = {};
  Object.keys(WORKFLOW_ZH).forEach((key) => { counts[key] = 0; });
  state.alerts.forEach((item) => { counts[item.workflow_status] = (counts[item.workflow_status] || 0) + 1; });
  Object.entries(counts).forEach(([key, value]) => {
    const row = el("div", "bar-row");
    row.append(el("span", "bar-label", WORKFLOW_ZH[key] || key), barTrack(key, value, state.alerts.length), el("span", "bar-label", String(value)));
    workflow.append(row);
  });
  const closure = el("div", "card");
  closure.append(el("h3", "card-title", "现场闭环"));
  closure.append(el("p", "", `已保存现场快照：${p.observation_snapshots} 份`));
  closure.append(el("p", "", `开放任务：${p.open_tasks} 个`));
  closure.append(el("p", "notice green", "现场快照不可变，且不会修改冻结模型的预测、显著性水平与告警结论。"));
  lower.append(workflow, closure);
  root.append(lower);
}

function distributionCard(title, counts, order, colors, nameFn) {
  const card = el("div", "card");
  card.append(el("h3", "card-title", title));
  const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
  const entries = order || Object.entries(counts).sort((a, b) => b[1] - a[1]);
  entries.forEach((entry) => {
    const key = Array.isArray(entry) ? entry[0] : entry;
    const label = Array.isArray(entry) && entry[1] ? entry[1] : (nameFn ? nameFn(key) : key);
    const value = counts[key] || 0;
    const row = el("div", "bar-row");
    row.append(el("span", "bar-label", label), barTrack(key, value, total, colors), el("span", "bar-label", String(value)));
    card.append(row);
  });
  return card;
}

function barTrack(key, value, total, colors) {
  const track = el("div", "bar-track");
  const fill = el("div", "bar-fill");
  fill.style.width = `${Math.round(100 * value / total)}%`;
  fill.style.background = colors ? (colors[key] || "var(--brand)") : "var(--brand)";
  track.append(fill);
  return track;
}

/* ================= 告警工作台 ================= */
function renderAlerts() {
  setTitle("告警工作台", "按等级、国家、作物与工作流状态筛选田块");
  const root = $("viewRoot");
  root.replaceChildren();
  const filter = el("div", "filter-bar");
  const level = selectField("等级", [["", "全部等级"], ["action", "行动告警"], ["watch", "关注复查"], ["normal", "常规监测"]], state.filters.level);
  const country = selectField("国家", [["", "全部国家"], ...Object.entries(COUNTRY_ZH)], state.filters.country);
  const crop = selectField("作物", [["", "全部作物"], ...Object.entries(CROP_ZH)], state.filters.crop);
  const status = selectField("状态", [["", "全部状态"], ...Object.entries(WORKFLOW_ZH)], state.filters.status);
  const search = el("input");
  search.placeholder = "搜索田块编号";
  search.value = state.filters.q;
  search.addEventListener("input", () => { state.filters.q = search.value.trim().toLowerCase(); renderAlertList(); });
  [level, country, crop, status].forEach((ctrl) => ctrl.addEventListener("change", () => {
    state.filters[ctrl.name] = ctrl.value;
    renderAlertList();
  }));
  filter.append(level, country, crop, status, search, el("span", "spacer"), el("span", "badge brand", `${filteredAlerts().length} 个田块`));
  root.append(filter);
  root.append(el("div", "list"));
  renderAlertList();
}

function selectField(name, options, current) {
  const select = el("select");
  select.name = name;
  options.forEach(([value, label]) => {
    const option = el("option", "", label);
    option.value = value;
    if (value === current) option.selected = true;
    select.append(option);
  });
  return select;
}

function filteredAlerts() {
  return state.alerts.filter((item) => {
    if (state.filters.level && item.risk_level !== state.filters.level) return false;
    if (state.filters.country && item.country !== state.filters.country) return false;
    if (state.filters.crop && item.crop !== state.filters.crop) return false;
    if (state.filters.status && item.workflow_status !== state.filters.status) return false;
    if (state.filters.q && !item.field_id.toLowerCase().includes(state.filters.q)) return false;
    return true;
  });
}

function renderAlertList() {
  const container = $("viewRoot").querySelector(".list");
  container.replaceChildren();
  const items = filteredAlerts().slice(0, state.filters.limit);
  if (!items.length) { container.append(el("div", "empty", "没有匹配的田块")); return; }
  items.forEach((item) => {
    const card = el("div", `alert-card level-${item.risk_level}`);
    const meta = el("div", "meta");
    meta.append(el("span", "title", `${countryZh(item.country)} · ${cropZh(item.crop)} · ${item.season} 季 · 田块 ${item.field_id.split("_").pop().split("-")[0]}`));
    const chips = el("div", "chips");
    chips.append(el("span", "chip brand", `观测至第 ${item.current_look} 期`));
    chips.append(el("span", "chip", WORKFLOW_ZH[item.workflow_status] || item.workflow_status));
    chips.append(el("span", "chip", `原始口径：${item.action_original ? "行动" : item.watch_original ? "关注" : "常规"}`));
    meta.append(chips);
    meta.append(el("span", "sub mono", item.field_id));
    const right = el("div", "right");
    right.append(badge(item.risk_level));
    right.append(el("span", "button", "查看详情"));
    card.append(meta, right);
    card.addEventListener("click", () => openAlert(item.field_id));
    container.append(card);
  });
}

/* ================= 报告中心 ================= */
function fieldShortName(fieldId) {
  const match = String(fieldId).match(/^([A-Za-z]+)_DUP\d+_farm\d+_field(\d+)_([a-z]+)_(\d{4})$/);
  if (!match) return fieldId;
  return `${countryZh(match[1])} · ${cropZh(match[3])} · ${match[4]} 季 · 田块 ${match[2]}`;
}

function renderReports() {
  setTitle("报告中心", "全部生成报告 · 按方式、状态、地区筛选与导出");
  const root = $("viewRoot");
  root.replaceChildren();
  if (!state.reports) { root.append(el("div", "empty", "数据加载失败")); return; }
  const filter = el("div", "filter-bar");
  const generator = selectField("generator", [["", "全部生成方式"], ["deterministic_fallback", "规则生成"], ["llm_guarded", "智能润色"]], state.reportFilters.generator);
  const status = selectField("status", [["", "全部状态"], ["needs_review", "待审核"], ["approved", "已批准"], ["rejected", "已驳回"], ["needs_revision", "待修改"]], state.reportFilters.status);
  const country = selectField("country", [["", "全部国家"], ...Object.entries(COUNTRY_ZH)], state.reportFilters.country);
  const crop = selectField("crop", [["", "全部作物"], ...Object.entries(CROP_ZH)], state.reportFilters.crop);
  const search = el("input");
  search.placeholder = "搜索田块编号";
  search.value = state.reportFilters.q;
  [generator, status, country, crop].forEach((ctrl) => ctrl.addEventListener("change", () => {
    state.reportFilters[ctrl.name] = ctrl.value;
    renderReportTable();
  }));
  search.addEventListener("input", () => {
    state.reportFilters.q = search.value.trim().toLowerCase();
    renderReportTable();
  });
  filter.append(generator, status, country, crop, search, el("span", "spacer"),
    el("span", "badge brand", `${filteredReports().length} 份报告`));
  root.append(filter);
  const wrap = el("div", "table-wrap");
  wrap.id = "reportTableWrap";
  root.append(wrap);
  renderReportTable();
}

function filteredReports() {
  const f = state.reportFilters;
  return state.reports.reports.filter((row) => {
    if (f.generator && row.generator !== f.generator) return false;
    if (f.status && row.status !== f.status) return false;
    if (f.country && row.country !== f.country) return false;
    if (f.crop && row.crop !== f.crop) return false;
    if (f.q && !row.event_key.toLowerCase().includes(f.q)) return false;
    return true;
  });
}

function renderReportTable() {
  const wrap = $("reportTableWrap");
  if (!wrap) return;
  wrap.replaceChildren();
  const table = el("table");
  const thead = el("thead");
  const head = el("tr");
  ["田块", "生成时间", "生成方式", "引擎", "状态", "操作"].forEach((h) => head.append(el("th", "", h)));
  thead.append(head);
  const body = el("tbody");
  filteredReports().slice(0, 300).forEach((row) => {
    const tr = el("tr");
    const nameCell = el("td");
    nameCell.append(el("span", "", fieldShortName(row.event_key)), el("div", "sub mono", row.event_key));
    const actionCell = el("td");
    const viewButton = el("a", "button", "查看");
    viewButton.href = "#";
    viewButton.addEventListener("click", (event) => {
      event.preventDefault();
      openReportFromCenter(row.event_key);
    });
    const mdLink = el("a", "button", "文档");
    mdLink.href = `/api/v2/export/report/${encodeURIComponent(row.event_key)}.md`;
    mdLink.target = "_blank";
    mdLink.rel = "noopener";
    const pdfLink = el("a", "button primary", "PDF");
    pdfLink.href = `/api/v2/export/report/${encodeURIComponent(row.event_key)}.pdf`;
    pdfLink.target = "_blank";
    pdfLink.rel = "noopener";
    actionCell.append(viewButton, mdLink, pdfLink);
    tr.append(
      nameCell,
      el("td", "", row.created_at.slice(0, 19).replace("T", " ")),
      el("td", "", badgeText(GENERATOR_ZH[row.generator] || row.generator, row.generator)),
      el("td", "", PROVIDER_ZH[row.provider] || row.provider),
      el("td", "", badgeText(REPORT_STATUS_ZH[row.status] || row.status, row.status)),
      actionCell,
    );
    body.append(tr);
  });
  table.append(thead, body);
  wrap.append(table);
}

function badgeText(text, key) {
  const badge = el("span", `badge ${key === "llm_guarded" ? "brand" : key === "approved" ? "brand" : key === "needs_review" ? "warn" : "muted"}`, text);
  return badge;
}

async function openReportFromCenter(fieldId) {
  switchView("alerts");
  await openAlert(fieldId);
  state.detailTab = "report";
  renderDrawer();
}

/* ================= 田块详情抽屉 ================= */
async function openAlert(fieldId) {
  showDrawer(null, "正在加载田块详情…");
  try {
    const detail = await api(`/api/v2/alerts/${encodeURIComponent(fieldId)}`);
    state.detail = detail;
    state.detailTab = "evidence";
    renderDrawer();
  } catch (error) {
    hideDrawer();
    toast(error.message, true);
  }
}

function showDrawer(children, title) {
  const drawer = $("drawer");
  drawer.replaceChildren();
  if (title) drawer.append(el("h2", "", title));
  if (children) drawer.append(children);
  drawer.classList.remove("hidden");
  $("drawerBackdrop").classList.remove("hidden");
}

function hideDrawer() {
  $("drawer").classList.add("hidden");
  $("drawerBackdrop").classList.add("hidden");
  state.detail = null;
}

function renderDrawer() {
  const detail = state.detail;
  const evidence = detail.evidence;
  const identity = evidence.identity;
  const model = evidence.model_evidence;
  const boundary = evidence.current_information_boundary;
  const level = model.action ? "action" : (model.watch ? "watch" : "normal");

  const head = el("div", "drawer-head");
  const titleBox = el("div");
  titleBox.append(el("h2", "", displayName(identity)));
  titleBox.append(el("div", "sub", `农场 ${identity.farm_id.split("_farm").pop()} · 田块编号 ${identity.field_number}`));
  titleBox.append(el("div", "mono", identity.field_id));
  const actions = el("div", "topbar-actions");
  actions.append(badge(level));
  actions.append(el("span", "badge brand", `决策口径：修正口径`));
  const close = el("button", "button ghost", "关闭");
  close.addEventListener("click", hideDrawer);
  actions.append(close);
  head.append(titleBox, actions);

  const info = el("div", "detail-grid");
  const cells = [
    ["当前观测期", `第 ${boundary.look} 期 · 积温 ${fmt(boundary.gdd_cutoff, 0)}`, ""],
    ["数据截止日期", boundary.cutoff_date || "—", ""],
    ["预测产量", `${fmt(model.prediction_t_ha, 3)} 吨/公顷`, model.prediction_t_ha < model.training_defined_low_yield_cutoff_t_ha ? "strong" : ""],
    ["低产阈值", `${fmt(model.training_defined_low_yield_cutoff_t_ha, 3)} 吨/公顷`, ""],
    ["越界检验 p 值", fmt(model.null_pvalue, 4), model.null_pvalue <= model.look_alpha ? "strong" : ""],
    ["观测期显著性水平", fmt(model.look_alpha, 4), ""],
    ["90% / 95% 上界", `${fmt(evidence.uncertainty.upper_90, 2)} / ${fmt(evidence.uncertainty.upper_95, 2)} 吨/公顷`, ""],
    ["原始冻结口径", model.action_original ? "行动告警" : model.watch_original ? "关注" : "常规", ""],
  ];
  cells.forEach(([k, v, cls]) => {
    const cell = el("div", "detail-cell");
    cell.append(el("div", "k", k), el("div", `v${cls ? ` ${cls}` : ""}`, String(v)));
    info.append(cell);
  });

  const tabs = el("div", "tabs");
  [
    ["evidence", "预警证据"], ["report", "中文报告"], ["field", "现场闭环"],
    ["tasks", "任务与状态"], ["timeline", "时间线"], ["versions", "报告版本"],
  ].forEach(([key, label]) => {
    const tab = el("button", `tab${state.detailTab === key ? " active" : ""}`, label);
    tab.addEventListener("click", () => { state.detailTab = key; renderDrawer(); });
    tabs.append(tab);
  });

  const body = el("div", "drawer-body");
  if (state.detailTab === "evidence") body.append(evidencePanel(evidence));
  if (state.detailTab === "report") body.append(reportPanel(detail, evidence));
  if (state.detailTab === "field") body.append(fieldPanel());
  if (state.detailTab === "tasks") body.append(tasksPanel());
  if (state.detailTab === "timeline") body.append(timelinePanel());
  if (state.detailTab === "versions") body.append(versionsPanel());

  const drawer = $("drawer");
  drawer.replaceChildren(head, info, tabs, body);
}

function evidencePanel(evidence) {
  const wrap = el("div");
  const chart = el("div", "card");
  chart.append(el("h3", "card-title", "三个观测期的越界证据", el("span", "card-sub", "柱高为 p 值，橙线为该期显著性水平")));
  chart.append(pvalueChart(evidence.look_history));
  wrap.append(chart);

  const tableCard = el("div", "card");
  tableCard.append(el("h3", "card-title", "逐观测期决策明细"));
  const tableWrap = el("div", "table-wrap");
  const table = el("table");
  const head = el("tr");
  ["观测期", "截止日期", "积温阈值", "预测产量（吨/公顷）", "低产阈值", "p 值", "显著性水平", "原始关注", "修正关注", "行动告警"].forEach((h) => head.append(el("th", "num", h)));
  const thead = el("thead");
  thead.append(head);
  table.append(thead);
  const tbody = el("tbody");
  evidence.look_history.forEach((row) => {
    const tr = el("tr");
    [row.look, row.cutoff_date || "—", fmt(row.gdd_cutoff, 0), fmt(row.prediction_t_ha, 3), fmt(row.low_yield_cutoff_t_ha, 3),
      fmt(row.null_pvalue, 4), fmt(row.alpha, 4), row.watch_original ? "关注" : "—", row.watch_clean ? "关注" : "—",
      row.look >= 2 && evidence.model_evidence.action ? "行动" : "—",
    ].forEach((v) => tr.append(el("td", "num", String(v))));
    tbody.append(tr);
  });
  table.append(tbody);
  tableWrap.append(table);
  tableCard.append(tableWrap);
  wrap.append(tableCard);

  const limits = el("div", "card");
  limits.append(el("h3", "card-title", "数据质量与解释边界"));
  evidence.data_quality.limitations.forEach((item) => limits.append(el("p", "", `· ${item}`)));
  wrap.append(limits);
  return wrap;
}

function pvalueChart(history) {
  const present = history.filter((row) => !row.missing);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  const width = 720, height = 240, margin = { left: 54, right: 18, top: 18, bottom: 38 };
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("style", "width:100%;height:auto");
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const maxY = 0.05;
  const x = (index) => margin.left + (present.length === 1 ? innerW / 2 : index * innerW / (present.length - 1));
  const y = (value) => margin.top + innerH - value / maxY * innerH;
  [0.01, 0.02, 0.03, 0.04].forEach((tick) => {
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", margin.left); line.setAttribute("y1", y(tick));
    line.setAttribute("x2", margin.left + innerW); line.setAttribute("y2", y(tick));
    line.setAttribute("stroke", "#e3e9e5"); line.setAttribute("stroke-width", "1");
    svg.append(line);
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", margin.left - 8); label.setAttribute("y", y(tick) + 4);
    label.setAttribute("text-anchor", "end"); label.setAttribute("font-size", "10"); label.setAttribute("fill", "#8b9a90");
    label.textContent = tick.toFixed(2);
    svg.append(label);
  });
  const alphaLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
  alphaLine.setAttribute("x1", margin.left); alphaLine.setAttribute("y1", y(0.01));
  alphaLine.setAttribute("x2", margin.left + innerW); alphaLine.setAttribute("y2", y(0.01));
  alphaLine.setAttribute("stroke", "#b26a1a"); alphaLine.setAttribute("stroke-width", "1.5");
  alphaLine.setAttribute("stroke-dasharray", "6 4");
  svg.append(alphaLine);
  present.forEach((row, index) => {
    const cx = x(index), cy = y(row.null_pvalue);
    const bar = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    bar.setAttribute("x", cx - 14); bar.setAttribute("y", cy);
    bar.setAttribute("width", 28); bar.setAttribute("height", Math.max(1, margin.top + innerH - cy));
    bar.setAttribute("rx", 5);
    bar.setAttribute("fill", row.watch_clean ? "#2d6da3" : "#9db8ac");
    svg.append(bar);
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", cx); text.setAttribute("y", height - 14); text.setAttribute("text-anchor", "middle");
    text.setAttribute("font-size", "11"); text.setAttribute("fill", "#5b6b62");
    text.textContent = `第 ${row.look} 期`;
    svg.append(text);
    if (row.watch_clean) {
      const flag = document.createElementNS("http://www.w3.org/2000/svg", "text");
      flag.setAttribute("x", cx); flag.setAttribute("y", cy - 8); flag.setAttribute("text-anchor", "middle");
      flag.setAttribute("font-size", "11"); flag.setAttribute("fill", "#2d6da3");
      flag.textContent = "越界";
      svg.append(flag);
    }
  });
  return svg;
}

function reportPanel(detail, evidence) {
  const wrap = el("div");
  const report = detail.report;
  if (!report) {
    const empty = el("div", "card");
    empty.append(el("h3", "card-title", "报告尚未生成"));
    empty.append(el("p", "", "点击下方按钮生成本田块的专业中文报告；可选智能润色，失败自动回退规则版本。"));
    wrap.append(empty);
  } else {
    const header = el("div", "card");
    header.append(el("div", "chips", ""),
      el("span", "badge brand", REPORT_STATUS_ZH[report.status] || report.status),
      el("span", "badge muted", GENERATOR_ZH[report.provenance.generator] || report.provenance.generator),
      el("span", "badge muted", `报告编号 ${report.report_id.slice(-12)}`),
      el("span", "badge muted", `生成时间 ${(report.generated_at || "").slice(0, 19).replace("T", " ")}`));
    wrap.append(header);
    const tabs = el("div", "tabs");
    const tabDefs = [["farmer", "农户摘要"], ["professional", "专业分析"], ["audit", "审计信息"]];
    if (detail.compare && (detail.compare.rule || detail.compare.llm)) tabDefs.push(["compare", "润色对比"]);
    tabDefs.forEach(([key, label]) => {
      const tab = el("button", `tab${state.reportTab === key ? " active" : ""}`, label);
      tab.addEventListener("click", () => { state.reportTab = key; renderDrawer(); });
      tabs.append(tab);
    });
    wrap.append(tabs);
    if (state.reportTab === "farmer") wrap.append(farmerReport(report));
    if (state.reportTab === "professional") wrap.append(professionalReport(report));
    if (state.reportTab === "audit") wrap.append(auditReport(report));
    if (state.reportTab === "compare") wrap.append(compareReport(detail.compare, evidence));
  }

  const progress = el("div", "card");
  progress.id = "genProgress";
  progress.classList.add("hidden");
  wrap.append(progress);

  const controls = el("div", "card");
  controls.append(el("h3", "card-title", "报告生成与审核"));
  const row = el("div", "filter-bar");
  const provider = el("select");
  Object.entries(PROVIDER_ZH).forEach(([value, label]) => {
    const option = el("option", "", label);
    option.value = value;
    provider.append(option);
  });
  provider.value = state.provider;
  const generate = el("button", "button primary", "生成报告");
  generate.addEventListener("click", (ev) => generateReport(evidence.identity.field_id, provider.value, ev.target));
  const printButton = el("button", "button", "打印报告");
  printButton.disabled = !report;
  printButton.addEventListener("click", () => { if (report) printReport(report); });
  const exportLink = el("a", "button", "导出文档");
  exportLink.href = `/api/v2/export/report/${encodeURIComponent(evidence.identity.field_id)}.md`;
  exportLink.target = "_blank";
  exportLink.rel = "noopener";
  const pdfLink = el("a", "button primary", "下载 PDF");
  pdfLink.href = `/api/v2/export/report/${encodeURIComponent(evidence.identity.field_id)}.pdf`;
  pdfLink.target = "_blank";
  pdfLink.rel = "noopener";
  row.append(el("span", "", "生成引擎"), provider, generate, printButton, exportLink, pdfLink);
  controls.append(row);

  const reviewRow = el("div", "filter-bar");
  const comment = el("input");
  comment.placeholder = "审核意见（驳回或退回必须填写）";
  comment.style.flex = "1";
  comment.style.minWidth = "220px";
  const approve = el("button", "button primary", "批准发布");
  approve.disabled = !report;
  approve.addEventListener("click", () => reviewReport(report, "approved", comment.value));
  const reject = el("button", "button danger", "驳回");
  reject.disabled = !report;
  reject.addEventListener("click", () => reviewReport(report, "rejected", comment.value));
  const revise = el("button", "button", "退回修改");
  revise.disabled = !report;
  revise.addEventListener("click", () => reviewReport(report, "needs_revision", comment.value));
  reviewRow.append(el("span", "", "人工审核"), comment, approve, reject, revise);
  controls.append(reviewRow);
  wrap.append(controls);
  return wrap;
}

function farmerReport(report) {
  const card = el("div", "card");
  card.append(el("h3", "", report.farmer_summary.headline));
  card.append(el("p", "", report.farmer_summary.what_we_observed));
  card.append(el("strong", "", "现在可以做什么"));
  report.farmer_summary.what_to_do_now.forEach((item) => card.append(el("p", "", `· ${item}`)));
  card.append(el("p", "notice green", report.farmer_summary.important_note));
  return card;
}

function professionalReport(report) {
  const card = el("div", "card");
  const p = report.professional_analysis;
  card.append(el("h3", "", p.conclusion));
  card.append(el("p", "", p.narrative));
  card.append(el("h4", "", "观测轨迹解读"));
  card.append(el("p", "", p.trajectory_assessment.narrative));
  card.append(el("h4", "", "空间证据"));
  card.append(el("p", "", p.spatial_assessment.narrative));
  card.append(el("h4", "", "锁定证据项"));
  p.evidence_items.forEach((item) => {
    card.append(el("p", "", `· ${item.title}：${String(item.value)}（${item.limitation}）`));
  });
  card.append(el("h4", "", "分阶段处置计划"));
  card.append(el("p", "", report.action_plan.rationale));
  [["within_24_hours", "24 小时内"], ["within_24_to_72_hours", "24 至 72 小时"], ["after_confirmation", "现场确认后"], ["do_not_do", "明确不要做"]].forEach(([key, label]) => {
    const items = report.action_plan[key];
    if (items && items.length) {
      card.append(el("strong", "", label));
      items.forEach((item) => card.append(el("p", "", `· ${item}`)));
    }
  });
  card.append(el("h4", "", "复查与升级"));
  card.append(el("p", "", `建议窗口：${report.recheck_plan.recommended_window}`));
  card.append(el("p", "", `降级条件：${report.recheck_plan.close_condition}`));
  card.append(el("p", "", `升级条件：${report.recheck_plan.escalate_condition}`));
  card.append(el("h4", "", "局限性与来源"));
  report.limitations.forEach((item) => card.append(el("p", "", `· ${item}`)));
  report.knowledge_citations.forEach((source) => {
    const link = el("a", "", `· ${source.title}`);
    link.href = source.url; link.target = "_blank"; link.rel = "noopener noreferrer";
    card.append(link);
  });
  return card;
}

function auditReport(report) {
  const card = el("div", "card");
  card.append(el("h3", "card-title", "报告审计信息"));
  const rows = [
    ["报告编号", report.report_id],
    ["状态", REPORT_STATUS_ZH[report.status] || report.status],
    ["生成方式", GENERATOR_ZH[report.provenance.generator] || report.provenance.generator],
    ["模型", report.provenance.model],
    ["证据哈希", report.provenance.evidence_hash],
    ["知识库版本", report.provenance.knowledge_version],
    ["决策口径", "修正口径（仅校准参考场）"],
  ];
  rows.forEach(([k, v]) => {
    const row = el("p", "");
    row.append(el("strong", "", `${k}：`), el("span", "mono", String(v)));
    card.append(row);
  });
  card.append(el("p", "notice", "报告在人工批准前不得作为正式处置指令。"));
  return card;
}

function highlightAdditions(text) {
  const node = el("span");
  const value = String(text || "");
  const marker = "补充解释：";
  const index = value.indexOf(marker);
  if (index < 0) {
    node.textContent = value;
    return node;
  }
  node.append(document.createTextNode(value.slice(0, index)));
  const add = el("span", "diff-add", value.slice(index));
  node.append(add);
  return node;
}

function compareReport(compare, evidence) {
  const wrap = el("div");
  if (!compare.llm) {
    const card = el("div", "card");
    card.append(el("h3", "card-title", "智能润色对比"));
    card.append(el("p", "", "当前只有规则版本。点击下方按钮用大模型生成润色版，即可并排对比：规则保证事实，润色优化表达。"));
    const button = el("button", "button primary", "生成智能润色版");
    button.addEventListener("click", (ev) => generateReport(evidence.identity.field_id, "deepseek", ev.target));
    card.append(button);
    wrap.append(card);
    return wrap;
  }
  const rule = compare.rule;
  const llm = compare.llm;
  const header = el("div", "card");
  header.append(el("h3", "card-title", "智能润色对比", el("span", "card-sub", "绿色为润色新增/改写，左侧为规则原文")));
  const legend = el("div", "chips");
  legend.append(el("span", "chip brand", "规则版"), el("span", "chip", "润色版"), el("span", "chip", "润色新增"));
  header.append(legend);
  wrap.append(header);

  const rows = [
    ["农户标题", rule.farmer_summary.headline, llm.farmer_summary.headline, false],
    ["农户提示", rule.farmer_summary.important_note, llm.farmer_summary.important_note, false],
    ["专业结论", rule.professional_analysis.conclusion, llm.professional_analysis.conclusion, false],
    ["专业叙事", rule.professional_analysis.narrative, llm.professional_analysis.narrative, true],
    ["行动理由", rule.action_plan.rationale, llm.action_plan.rationale, true],
  ];
  rows.forEach(([label, ruleText, llmText, highlight]) => {
    const card = el("div", "card");
    const changed = ruleText !== llmText;
    card.append(el("h3", "card-title", label, el("span", "card-sub", changed ? "润色版有改动" : "两版一致")));
    const grid = el("div", "compare-grid");
    const ruleBox = el("div", "compare-box");
    ruleBox.append(el("div", "compare-label", "规则版"));
    ruleBox.append(el("p", "", ruleText));
    const llmBox = el("div", "compare-box llm");
    llmBox.append(el("div", "compare-label", "智能润色版"));
    const content = highlight ? highlightAdditions(llmText) : el("p", "", llmText);
    llmBox.append(content);
    grid.append(ruleBox, llmBox);
    card.append(grid);
    wrap.append(card);
  });
  return wrap;
}

const STAGE_ZH = {
  queued: "任务排队", preparing: "准备证据", calling_llm: "调用大模型",
  merging: "合并叙述", validating: "事实校验", fallback: "安全回退",
  done: "完成", failed: "失败",
};

function renderGenerateProgress(job) {
  const container = $("genProgress");
  if (!container) return;
  container.classList.remove("hidden");
  container.replaceChildren();
  container.append(el("h3", "card-title", "生成进度"));
  const events = job.events || [];
  if (!events.length) { container.append(el("p", "muted", "等待任务执行…")); return; }
  const list = el("div", "timeline");
  events.forEach((event) => {
    const row = el("div", "timeline-item");
    const state = event.stage === "failed" ? "✕" : event.stage === "done" || event.stage === "fallback" ? "✓" : "●";
    const stateClass = event.stage === "failed" ? "stage-fail" : event.stage === "done" ? "stage-done" : "stage-run";
    row.append(el("span", `stage-dot ${stateClass}`, state));
    const text = el("div");
    text.append(el("strong", "", `${STAGE_ZH[event.stage] || event.stage}：${event.message}`));
    text.append(el("div", "sub", (event.at || "").slice(11, 19)));
    row.append(text);
    list.append(row);
  });
  container.append(list);
}

async function generateReport(fieldId, provider, button) {
  button.disabled = true;
  button.textContent = "正在生成…";
  const progress = $("genProgress");
  if (progress) progress.classList.remove("hidden");
  try {
    const job = await api("/api/v2/reports/generate", {
      method: "POST",
      body: JSON.stringify({ event_key: fieldId, provider, idempotency_key: `ui-${Date.now()}` }),
    });
    renderGenerateProgress(job);
    let result = null;
    for (let i = 0; i < 120; i++) {
      const status = await api(`/api/v2/jobs/${job.job_id}`);
      renderGenerateProgress(status);
      if (status.status === "done") { result = status.result; break; }
      if (status.status === "failed") throw new Error(status.error);
      await new Promise((resolve) => setTimeout(resolve, 150));
    }
    if (!result) throw new Error("生成超时，请稍后重试");
    state.detail = await api(`/api/v2/alerts/${encodeURIComponent(fieldId)}`);
    state.detailTab = "report";
    state.reportTab = "farmer";
    toast(result.warning || "报告已生成并通过事实校验，等待人工审核", Boolean(result.warning));
    renderDrawer();
    await refresh();
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "生成报告";
  }
}

async function reviewReport(report, decision, comment) {
  if (!report) return;
  if (decision !== "approved" && !comment.trim()) { toast("驳回或退回必须填写说明", true); return; }
  try {
    await api(`/api/v2/reports/${encodeURIComponent(report.report_id)}/review`, {
      method: "POST",
      body: JSON.stringify({ decision, comment }),
    });
    toast(`审核状态已更新：${REPORT_STATUS_ZH[decision] || decision}`);
    await openAlert(report.event_key);
    await refresh();
  } catch (error) {
    toast(error.message, true);
  }
}

function fieldPanel() {
  const wrap = el("div");
  const form = el("div", "card field-form");
  form.append(el("h3", "card-title", "提交现场观察快照", el("span", "card-sub", "不可变，不改变模型结论")));
  const moisture = el("select");
  ["未记录", "偏干", "适中", "偏湿"].forEach((v) => moisture.append(new Option(v, v)));
  const water = el("select");
  ["未记录", "未见", "局部", "明显"].forEach((v) => water.append(new Option(v, v)));
  const symptoms = el("input"); symptoms.placeholder = "如：叶色略淡、局部萎蔫待复核";
  const notes = el("textarea"); notes.rows = 2; notes.placeholder = "位置、照片编号、农事记录等";
  const save = el("button", "button primary", "保存快照");
  save.addEventListener("click", async () => {
    try {
      await api(`/api/v2/alerts/${encodeURIComponent(state.detail.evidence.identity.field_id)}/observations`, {
        method: "POST",
        body: JSON.stringify({
          soil_moisture: moisture.value, standing_water: water.value,
          canopy_symptoms: symptoms.value, notes: notes.value,
        }),
      });
      toast("现场快照已保存（不可变，不改变模型结论）");
      await openAlert(state.detail.evidence.identity.field_id);
    } catch (error) { toast(error.message, true); }
  });
  [["土壤水分", moisture], ["田间积水", water]].forEach(([label, control]) => {
    const box = el("label", "", label);
    box.append(control);
    form.append(box);
  });
  form.append(labelBox("冠层症状", symptoms), labelBox("补充说明", notes), save);
  wrap.append(form);
  const list = el("div", "card");
  list.append(el("h3", "card-title", `现场快照历史（${state.detail.snapshots.length} 份）`));
  if (!state.detail.snapshots.length) {
    list.append(el("p", "empty", "暂无现场快照"));
  } else {
    state.detail.snapshots.forEach((snapshot) => {
      const item = el("div", "timeline-item");
      item.append(el("time", "", snapshot.created_at.slice(0, 16).replace("T", " ")));
      const text = el("div");
      text.append(el("strong", "", `快照 ${snapshot.snapshot_id.slice(0, 8)} · ${snapshot.observer}`));
      text.append(el("div", "sub", `土壤水分：${snapshot.soil_moisture || "未记录"}；积水：${snapshot.standing_water || "未记录"}`));
      item.append(text);
      list.append(item);
    });
  }
  wrap.append(list);
  return wrap;
}

function labelBox(label, control) {
  const box = el("label", "", label);
  box.append(control);
  return box;
}

function tasksPanel() {
  const wrap = el("div", "card");
  wrap.append(el("h3", "card-title", "任务与工作流"));
  const workflow = state.detail.workflow;
  wrap.append(el("p", "", `当前状态：${WORKFLOW_ZH[workflow.status] || workflow.status}${workflow.assignee ? ` · 负责人：${workflow.assignee}` : ""}${workflow.due_at ? ` · 截止：${workflow.due_at}` : ""}`));
  if (!state.detail.tasks.length) {
    wrap.append(el("p", "empty", "暂无任务（可通过接口分派现场核验）"));
  } else {
    const tableWrap = el("div", "table-wrap");
    const table = el("table");
    const head = el("tr");
    ["任务", "负责人", "截止时间", "状态"].forEach((h) => head.append(el("th", "", h)));
    const thead = el("thead");
    thead.append(head);
    table.append(thead);
    const body = el("tbody");
    state.detail.tasks.forEach((task) => {
      const tr = el("tr");
      [task.title, task.assignee, task.due_at, task.status].forEach((v) => tr.append(el("td", "", String(v))));
      body.append(tr);
    });
    table.append(body);
    tableWrap.append(table);
    wrap.append(tableWrap);
  }
  return wrap;
}

function timelinePanel() {
  const wrap = el("div", "card");
  wrap.append(el("h3", "card-title", "事件时间线"));
  const list = el("div", "timeline");
  state.detail.timeline.forEach((item) => {
    const row = el("div", "timeline-item");
    row.append(el("time", "", item.created_at.slice(0, 16).replace("T", " ")));
    const text = el("div");
    text.append(el("strong", "", item.title), el("p", "muted", item.detail || ""));
    row.append(text);
    list.append(row);
  });
  wrap.append(list);
  return wrap;
}

function versionsPanel() {
  const wrap = el("div", "card");
  wrap.append(el("h3", "card-title", `报告版本历史（${state.detail.report_versions.length} 条）`));
  if (!state.detail.report_versions.length) {
    wrap.append(el("p", "empty", "暂无报告版本"));
    return wrap;
  }
  const tableWrap = el("div", "table-wrap");
  const table = el("table");
  const head = el("tr");
  ["生成时间", "生成方式", "引擎", "模型", "状态", "编号"].forEach((h) => head.append(el("th", "", h)));
  const thead = el("thead");
  thead.append(head);
  table.append(thead);
  const body = el("tbody");
  state.detail.report_versions.forEach((row) => {
    const tr = el("tr");
    [row.created_at.slice(0, 19).replace("T", " "), GENERATOR_ZH[row.generator] || row.generator,
      PROVIDER_ZH[row.provider] || row.provider, row.model || "—",
      REPORT_STATUS_ZH[row.status] || row.status, row.report_id.slice(-12),
    ].forEach((v) => tr.append(el("td", "", String(v))));
    body.append(tr);
  });
  table.append(body);
  tableWrap.append(table);
  wrap.append(tableWrap);
  return wrap;
}

function printReport(report) {
  const p = report.professional_analysis;
  const html = `<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>低产风险报告 · ${esc(report.event_key)}</title>
<style>
  body{font-family:"PingFang SC","Microsoft YaHei",sans-serif;color:#1c2a22;margin:40px;line-height:1.7}
  h1{font-size:22px;margin-bottom:4px}.meta{color:#6a7a70;font-size:13px;margin-bottom:24px}
  h2{font-size:17px;border-left:4px solid #1e7a50;padding-left:10px;margin-top:28px}
  h3{font-size:14px}.notice{background:#f6f1e4;border:1px solid #eadfc0;padding:10px 14px;border-radius:8px;font-size:13px}
  .notice.green{background:#e8f3ed;border-color:#cfe5d8}table{border-collapse:collapse;width:100%;font-size:13px}
  th,td{border:1px solid #dfe6e1;padding:8px 10px;text-align:left}th{background:#f4f7f5}
  .pagebreak{page-break-before:always}
</style></head><body>
  <h1>低产风险预警报告</h1>
  <div class="meta">田块：${esc(report.event_key)} · 状态：${esc(REPORT_STATUS_ZH[report.status] || report.status)} · 生成方式：${esc(GENERATOR_ZH[report.provenance.generator] || report.provenance.generator)} · 生成时间：${esc((report.generated_at || "").slice(0, 19).replace("T", " "))}</div>
  <p><strong>结论：</strong>${esc(p.conclusion)}</p>
  <p>${esc(p.narrative)}</p>
  <h2>农户摘要</h2>
  <p><strong>${esc(report.farmer_summary.headline)}</strong></p>
  <p>${esc(report.farmer_summary.what_we_observed)}</p>
  <h3>现在可以做什么</h3>
  <ul>${report.farmer_summary.what_to_do_now.map((i) => `<li>${esc(i)}</li>`).join("")}</ul>
  <div class="notice green">${esc(report.farmer_summary.important_note)}</div>
  <h2>观测轨迹</h2>
  <p>${esc(p.trajectory_assessment.narrative)}</p>
  <h2>空间证据</h2>
  <p>${esc(p.spatial_assessment.narrative)}</p>
  <h2>锁定证据项</h2>
  <table><thead><tr><th>项目</th><th>数值</th><th>解释边界</th></tr></thead><tbody>
  ${p.evidence_items.map((i) => `<tr><td>${esc(i.title)}</td><td>${esc(i.value)}</td><td>${esc(i.limitation)}</td></tr>`).join("")}
  </tbody></table>
  <h2>分阶段处置计划</h2>
  <p>${esc(report.action_plan.rationale)}</p>
  <ul>${report.action_plan.within_24_hours.map((i) => `<li>${esc(i)}</li>`).join("")}</ul>
  <h3>明确不要做</h3>
  <ul>${report.action_plan.do_not_do.map((i) => `<li>${esc(i)}</li>`).join("")}</ul>
  <h2>复查与升级</h2>
  <p>建议窗口：${esc(report.recheck_plan.recommended_window)}</p>
  <p>降级条件：${esc(report.recheck_plan.close_condition)}</p>
  <p>升级条件：${esc(report.recheck_plan.escalate_condition)}</p>
  <h2>局限性与来源</h2>
  <ul>${report.limitations.map((i) => `<li>${esc(i)}</li>`).join("")}</ul>
  <p>证据哈希：<code>${esc(report.provenance.evidence_hash)}</code></p>
  <div class="notice">本报告在人工批准前不得作为正式处置指令；p 值不是低产概率，预测区间不是保证范围。</div>
</body></html>`;
  const win = window.open("", "_blank");
  if (!win) { toast("请允许弹出窗口以打印报告", true); return; }
  win.document.write(html);
  win.document.close();
  setTimeout(() => { win.print(); }, 300);
}

/* ================= 田块空间 ================= */
function renderFields() {
  setTitle("田块空间", "316 个开发田块的空间残差指标 · 仅用于田块内相对定位");
  const root = $("viewRoot");
  root.replaceChildren();
  if (!state.spatial) { root.append(el("div", "empty", "数据加载失败")); return; }
  root.append(el("p", "notice", state.spatial.disclosure));
  root.append(geojsonCard());
  const chartCard = el("div", "card");
  chartCard.append(el("h3", "card-title", "底部两成区域定位能力分布", el("span", "card-sub", "数值越高，田块内低产区域定位越准")));
  chartCard.append(histogram(state.spatial.fields.map((f) => f.bottom20_auroc)));
  root.append(chartCard);
  const tableCard = el("div", "card");
  tableCard.append(el("h3", "card-title", `空间指标明细（${state.spatial.count} 个田块）`));
  const search = el("input");
  search.placeholder = "搜索田块编号";
  search.style.marginBottom = "12px";
  const wrap = el("div", "table-wrap");
  function renderTable() {
    wrap.replaceChildren();
    const table = el("table");
    const head = el("tr");
    ["国家", "作物", "田块编号", "像素数", "残差均方根误差", "均匀分布均方根误差", "植被指数均方根误差", "决定系数", "秩相关", "底部两成定位能力"].forEach((h) => head.append(el("th", "num", h)));
    const thead = el("thead");
    thead.append(head);
    table.append(thead);
    const body = el("tbody");
    const query = search.value.trim().toLowerCase();
    state.spatial.fields.filter((f) => !query || f.field_id.toLowerCase().includes(query)).slice(0, 200).forEach((f) => {
      const tr = el("tr");
      [countryZh(f.country), cropZh(f.crop), f.field_id.split("_").pop().split("-")[0], f.pixels,
        fmt(f.rmse_spatial, 3), fmt(f.rmse_uniform, 3), fmt(f.rmse_ndvi, 3),
        fmt(f.r2_spatial, 3), fmt(f.rho_spatial, 3), fmt(f.bottom20_auroc, 3),
      ].forEach((v) => tr.append(el("td", "num", String(v))));
      body.append(tr);
    });
    table.append(body);
    wrap.append(table);
  }
  search.addEventListener("input", renderTable);
  renderTable();
  tableCard.append(search, wrap);
  root.append(tableCard);
}

function geojsonCard() {
  const card = el("div", "card");
  card.append(el("h3", "card-title", "田块边界文件预览", el("span", "card-sub", "支持 GeoJSON 面要素，仅前端展示，不参与决策")));
  const upload = el("div", "geojson-upload");
  const input = el("input");
  input.type = "file";
  input.accept = ".json,.geojson,application/geo+json";
  const hint = el("span", "hint", "可导入客户提供的真实田块边界，预览多边形与要素数量");
  const reset = el("button", "button", "恢复默认");
  reset.addEventListener("click", () => {
    input.value = "";
    const old = card.querySelector(".geojson-preview");
    if (old) old.remove();
  });
  input.addEventListener("change", () => {
    const file = input.files && input.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(String(reader.result));
        const features = (parsed.features || []).filter((f) =>
          ["Polygon", "MultiPolygon"].includes(f.geometry && f.geometry.type));
        if (!features.length) { toast("未找到面要素，请上传包含多边形的地块文件", true); return; }
        const old = card.querySelector(".geojson-preview");
        if (old) old.remove();
        card.append(geoPreview(features));
        toast(`已预览 ${features.length} 个地块边界`);
      } catch (error) {
        toast("文件解析失败，请确认是有效的 GeoJSON", true);
      }
    };
    reader.readAsText(file);
  });
  upload.append(input, hint, reset);
  card.append(upload);
  return card;
}

function geoPreview(features) {
  const wrap = el("div", "geojson-preview");
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 720 380");
  svg.setAttribute("style", "width:100%;height:auto");
  const points = [];
  features.forEach((feature) => {
    const rings = feature.geometry.type === "Polygon"
      ? feature.geometry.coordinates : feature.geometry.coordinates.flat();
    rings.forEach((ring) => ring.forEach((coord) => points.push(coord)));
  });
  if (!points.length) return wrap;
  const xs = points.map((c) => c[0]), ys = points.map((c) => c[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const spanX = Math.max(1e-9, maxX - minX), spanY = Math.max(1e-9, maxY - minY);
  const scale = Math.min(640 / spanX, 320 / spanY);
  const ox = 40 + (640 - spanX * scale) / 2, oy = 30 + (320 - spanY * scale) / 2;
  const px = (lon) => ox + (lon - minX) * scale;
  const py = (lat) => oy + (maxY - lat) * scale;
  features.slice(0, 120).forEach((feature, index) => {
    const rings = feature.geometry.type === "Polygon"
      ? feature.geometry.coordinates : feature.geometry.coordinates.flat();
    rings.forEach((ring) => {
      if (ring.length < 3) return;
      const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      polygon.setAttribute("points", ring.map((c) => `${px(c[0])},${py(c[1])}`).join(" "));
      polygon.setAttribute("fill", index % 2 ? "#bfe0cf" : "#8ccbad");
      polygon.setAttribute("stroke", "#1e7a50");
      polygon.setAttribute("stroke-width", "1.5");
      polygon.setAttribute("opacity", ".85");
      svg.append(polygon);
    });
  });
  wrap.append(svg, el("p", "muted", `共 ${features.length} 个地块边界（最多预览 120 个）`));
  return wrap;
}

function histogram(values) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 720 220");
  svg.setAttribute("style", "width:100%;height:auto");
  const bins = Array(10).fill(0);
  values.forEach((v) => { const i = Math.min(9, Math.floor(v * 10)); bins[i] += 1; });
  const max = Math.max(1, ...bins);
  const width = 660, height = 170, barW = width / bins.length;
  bins.forEach((count, i) => {
    const bar = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    bar.setAttribute("x", 30 + i * barW + 2);
    bar.setAttribute("y", height - count / max * height);
    bar.setAttribute("width", barW - 4);
    bar.setAttribute("height", count / max * height);
    bar.setAttribute("rx", 3);
    bar.setAttribute("fill", "#1e7a50");
    svg.append(bar);
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", 30 + i * barW + barW / 2);
    label.setAttribute("y", height + 18);
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("font-size", "10");
    label.setAttribute("fill", "#8b9a90");
    label.textContent = `0.${i}`;
    svg.append(label);
  });
  return svg;
}

/* ================= 现场任务 ================= */
function renderTasks() {
  setTitle("现场任务", "按工作流状态分组的田块队列");
  const root = $("viewRoot");
  root.replaceChildren();
  const kanban = el("div", "kanban");
  Object.entries(WORKFLOW_ZH).forEach(([status, label]) => {
    const col = el("div", "kanban-col");
    const items = state.alerts.filter((item) => item.workflow_status === status);
    col.append(el("h3", "", `${label}（${items.length}）`));
    items.slice(0, 14).forEach((item) => {
      const card = el("div", "kanban-card");
      card.append(el("strong", "", `${countryZh(item.country)} · ${cropZh(item.crop)} · 田块 ${item.field_id.split("_").pop().split("-")[0]}`));
      card.append(el("div", "sub", `${item.season} 季 · ${LEVEL_META[item.risk_level].label}${item.assignee ? ` · ${item.assignee}` : ""}`));
      card.addEventListener("click", () => openAlert(item.field_id));
      col.append(card);
    });
    kanban.append(col);
  });
  root.append(kanban);
}

/* ================= 知识库 ================= */
function renderKnowledge() {
  setTitle("知识库", "按国家与作物适用范围治理的已批准低风险信息采集条目");
  const root = $("viewRoot");
  root.replaceChildren();
  if (!state.knowledge) return;
  const meta = el("div", "card");
  meta.append(el("h3", "card-title", "治理信息"));
  meta.append(el("p", "", `版本：${state.knowledge.version} · 维护方：${state.knowledge.governance.owner}`));
  meta.append(el("p", "", `下次复核：${state.knowledge.governance.next_review_date}`));
  state.knowledge.governance.prohibited_advice.forEach((item) => meta.append(el("p", "", `· 禁止：${item}`)));
  root.append(meta);
  const grid = el("div", "grid grid-2");
  state.knowledge.entries.forEach((entry) => {
    const card = el("div", "card");
    card.append(el("h3", "card-title", entry.title));
    const chips = el("div", "chips");
    (entry.countries.length ? entry.countries.map(countryZh) : ["全部国家"]).forEach((c) => chips.append(el("span", "chip brand", c)));
    (entry.crops.length ? entry.crops.map(cropZh) : ["全部作物"]).forEach((c) => chips.append(el("span", "chip", c)));
    card.append(chips);
    card.append(el("strong", "", "低风险行动"));
    entry.low_risk_actions.forEach((item) => card.append(el("p", "", `· ${item}`)));
    if (entry.conditional_actions.length) {
      card.append(el("strong", "", "条件行动（需人工判断）"));
      entry.conditional_actions.forEach((item) => card.append(el("p", "", `· ${item}`)));
    }
    card.append(el("div", "sub", `来源：${entry.source_ids.join("、")}`));
    grid.append(card);
  });
  root.append(grid);
}

/* ================= 审计记录 ================= */
function renderAudit() {
  setTitle("审计记录", "操作日志与数据结构迁移记录");
  const root = $("viewRoot");
  root.replaceChildren();
  if (!state.audit) return;
  const card = el("div", "card");
  card.append(el("h3", "card-title", `操作日志（${state.audit.audit_log.length} 条）`));
  const wrap = el("div", "table-wrap");
  const table = el("table");
  const head = el("tr");
  ["编号", "操作者", "动作", "对象", "时间"].forEach((h) => head.append(el("th", "", h)));
  const thead = el("thead");
  thead.append(head);
  table.append(thead);
  const body = el("tbody");
  state.audit.audit_log.forEach((row) => {
    const tr = el("tr");
    [row.audit_id, row.actor, AUDIT_ACTION_ZH[row.action] || row.action, row.resource,
      row.created_at.slice(0, 19).replace("T", " "),
    ].forEach((v) => tr.append(el("td", "", String(v))));
    body.append(tr);
  });
  table.append(body);
  wrap.append(table);
  card.append(wrap);
  root.append(card);
  const migrations = el("div", "card");
  migrations.append(el("h3", "card-title", "数据结构迁移"));
  state.audit.schema_migrations.forEach((row) => migrations.append(el("p", "", `第 ${row.version} 版 · ${row.applied_at.slice(0, 19).replace("T", " ")}`)));
  root.append(migrations);
  if (state.genLogs && state.genLogs.logs.length) {
    const logs = el("div", "card");
    logs.append(el("h3", "card-title", `报告生成日志（${state.genLogs.count} 条）`));
    const tableWrap = el("div", "table-wrap");
    const table = el("table");
    const thead = el("thead");
    const head = el("tr");
    ["时间", "田块", "生成方式", "引擎 / 模型", "校验", "阶段摘要"].forEach((h) => head.append(el("th", "", h)));
    thead.append(head);
    const body = el("tbody");
    state.genLogs.logs.slice(0, 50).forEach((row) => {
      const tr = el("tr");
      const stages = row.stages || [];
      const summary = stages.map((s) => STAGE_ZH[s.stage] || s.stage).join(" → ");
      const passed = row.validation && row.validation.passes;
      tr.append(
        el("td", "", row.created_at.slice(0, 19).replace("T", " ")),
        el("td", "", fieldShortName(row.event_key)),
        el("td", "", GENERATOR_ZH[row.generator] || row.generator),
        el("td", "", `${PROVIDER_ZH[row.provider] || row.provider} / ${row.model}`),
        el("td", "", passed ? "通过" : "未通过（已回退）"),
        el("td", "", summary),
      );
      body.append(tr);
    });
    table.append(thead, body);
    tableWrap.append(table);
    logs.append(tableWrap);
    root.append(logs);
  }
}

/* ================= 回顾审计 ================= */
function renderRetrospective() {
  setTitle("回顾审计", "收获后评估 · 与前瞻告警严格隔离");
  const root = $("viewRoot");
  root.replaceChildren();
  if (!state.retrospective) return;
  root.append(el("p", "notice", state.retrospective.disclosure));
  const r = state.retrospective;
  const kpis = el("div", "grid grid-4");
  [
    ["审计状态", r.final_audit.status, "不可改写", "normal"],
    ["原始行动误报率", `${(r.final_audit.action.fpr * 100).toFixed(2)}%`, `命中 ${r.final_audit.action.tp} · 误报 ${r.final_audit.action.fp}`, "watch"],
    ["修正口径·关注", `误报率 ${(r.p0b_clean_null.watch.fpr * 100).toFixed(2)}%`, `敏感度 ${(r.p0b_clean_null.watch.sensitivity * 100).toFixed(2)}%`, "watch"],
    ["修正口径·行动", `误报率 ${(r.p0b_clean_null.action.fpr * 100).toFixed(2)}%`, `精确率 ${(r.p0b_clean_null.action.precision * 100).toFixed(2)}%`, "action"],
  ].forEach(([label, value, note, cls]) => {
    const kpi = el("div", `kpi ${cls}`);
    kpi.append(el("span", "kpi-label", label), el("strong", "kpi-value", value), el("span", "kpi-note", note));
    kpis.append(kpi);
  });
  root.append(kpis);
  const card = el("div", "card");
  card.append(el("h3", "card-title", `逐观测期收获后明细（${r.fields.length} 行）`));
  const search = el("input");
  search.placeholder = "搜索田块编号";
  search.style.marginBottom = "12px";
  const wrap = el("div", "table-wrap");
  function renderTable() {
    wrap.replaceChildren();
    const table = el("table");
    const head = el("tr");
    ["田块编号", "观测期", "预测产量（吨/公顷）", "实际产量（吨/公顷）", "真实低产", "p 值", "关注", "行动"].forEach((h) => head.append(el("th", "num", h)));
    const thead = el("thead");
    thead.append(head);
    table.append(thead);
    const body = el("tbody");
    const query = search.value.trim().toLowerCase();
    r.fields.filter((f) => !query || f.field_id.toLowerCase().includes(query)).slice(0, 200).forEach((f) => {
      const tr = el("tr");
      [f.field_id, f.look, fmt(f.prediction, 3), fmt(f.target, 3), f.true_low ? "是" : "否", fmt(f.pvalue, 4),
        f.watch ? "关注" : "—", f.action_alert ? "行动" : "—",
      ].forEach((v) => tr.append(el("td", "num", String(v))));
      body.append(tr);
    });
    table.append(body);
    wrap.append(table);
  }
  search.addEventListener("input", renderTable);
  renderTable();
  card.append(search, wrap);
  root.append(card);
}

/* ================= 价值评估 ================= */
function renderValue() {
  setTitle("价值评估", "把预警能力换算成可验证的巡田成本场景");
  const root = $("viewRoot");
  root.replaceChildren();

  const calc = el("div", "card");
  calc.append(el("h3", "card-title", "巡田成本场景计算器", el("span", "card-sub", "输入客户参数，现场算账")));
  const inputs = el("div", "calculator");
  const configs = [
    ["管理田块数", "calcFields", 1000],
    ["每季常规巡田次数", "calcVisits", 4],
    ["单地块单次巡田成本（元）", "calcCost", 120],
    ["预警后重点巡田比例（%）", "calcShare", 20],
  ];
  configs.forEach(([label, id, value]) => {
    const box = el("label", "", label);
    const input = el("input");
    input.id = id;
    input.type = "number";
    input.min = "1";
    input.value = String(value);
    input.addEventListener("input", renderRoi);
    box.append(input);
    inputs.append(box);
  });
  calc.append(inputs);
  const roi = el("div", "roi-grid");
  roi.id = "roiGrid";
  calc.append(roi);
  calc.append(el("p", "notice", "价值计算是用户可修改的情景估算，不是实测经济收益，不构成收益承诺。"));
  root.append(calc);

  const pilot = el("div", "card");
  pilot.append(el("h3", "card-title", "建议的 30 天企业试点"));
  const steps = el("div", "steps");
  [
    ["数据接入", "接入客户 1～2 个地区的田块边界与历史季节数据，完成前瞻回放。"],
    ["流程试跑", "农艺师团队使用告警工作台、现场快照与报告审核，记录真实使用反馈。"],
    ["成效评估", "按报告事实正确率、平均审核时长、巡田完成率、误升级/漏升级率出具试点报告。"],
    ["规模化决策", "依据试点指标决定按季节订阅的服务范围与田块数。"],
  ].forEach(([title, desc], index) => {
    const row = el("div", "step-row");
    row.append(el("span", "step-num", String(index + 1)));
    const text = el("div");
    text.append(el("h4", "", title), el("p", "", desc));
    row.append(text);
    steps.append(row);
  });
  pilot.append(steps);
  root.append(pilot);

  const moat = el("div", "card");
  moat.append(el("h3", "card-title", "为什么值得为这套系统付费"));
  const items = el("div", "moat-list");
  [
    ["可承诺的误报率", "行动告警误报率 1.6%、关注 4.8%（修正口径），每个告警都带显著性水平与审计证据。"],
    ["审计可重放", "每份报告绑定证据哈希、知识库版本与决策口径，可追溯到冻结实验文件。"],
    ["两级告警设计", "关注用于低成本巡田，行动用于高精度复核，直接服务巡田资源排布。"],
    ["人工把关闭环", "报告必须人工批准；现场快照不可变；系统绝不自动灌溉、施肥、施药。"],
    ["多国多作物", "阿根廷、巴西、德国、乌拉圭 × 玉米、大豆、小麦、油菜，框架可扩展。"],
    ["诚信记录", "一次性冻结审计与事后修正分开标注，失败如实保留，不粉饰结果。"],
  ].forEach(([title, desc]) => {
    const item = el("div", "moat-item");
    item.append(el("h4", "", title), el("p", "", desc));
    items.append(item);
  });
  moat.append(items);
  root.append(moat);
  renderRoi();
}

function renderRoi() {
  const fields = Math.max(0, Number($("calcFields").value) || 0);
  const visits = Math.max(0, Number($("calcVisits").value) || 0);
  const cost = Math.max(0, Number($("calcCost").value) || 0);
  const share = Math.max(0, Math.min(100, Number($("calcShare").value) || 0)) / 100;
  const baseline = fields * visits * cost;
  const targeted = baseline * share;
  const avoided = baseline - targeted;
  const format = new Intl.NumberFormat("zh-CN", { style: "currency", currency: "CNY", maximumFractionDigits: 0 });
  const grid = $("roiGrid");
  if (!grid) return;
  grid.replaceChildren();
  [
    ["常规全量巡田成本", format.format(baseline), "全部田块按现有频次巡田", ""],
    ["预警后重点巡田成本", format.format(targeted), `仅 ${Math.round(share * 100)}% 田块重点核验`, "brand"],
    ["可避免巡田支出", format.format(avoided), `节省比例 ${(share * 100).toFixed(0)}%（情景估算）`, "action"],
  ].forEach(([label, value, note, cls]) => {
    const card = el("div", "roi-card");
    card.append(el("div", "k", label), el("div", `v ${cls}`, value), el("div", "n", note));
    grid.append(card);
  });
}

/* ================= 系统设置 ================= */
function renderSettings() {
  setTitle("系统设置", "系统信息、数据源校验与可信边界");
  const root = $("viewRoot");
  root.replaceChildren();
  const card = el("div", "card");
  card.append(el("h3", "card-title", "数据源与完整性校验"));
  card.append(el("p", "", "证据由以下冻结审计产物重建：最终预测明细、修正口径对比、空间指标、积温里程碑、低产阈值参考、冻结清单、收获前提前量明细。"));
  card.append(el("p", "notice green", "输入文件哈希已固化；任何冻结文件被修改后，适配器会立即拒绝运行。"));
  root.append(card);
  const capability = el("div", "card");
  capability.append(el("h3", "card-title", "系统能力"));
  [
    "三级告警：行动告警 / 关注复查 / 常规监测",
    "三个观测期的越界证据与决策明细",
    "专业中文报告 + 智能润色（可完全关闭并自动回退）",
    "现场快照闭环与人工审核、版本历史",
    "田块空间定位与收获后回顾审计（隔离）",
    "操作审计与数据结构迁移记录",
  ].forEach((item) => capability.append(el("p", "", `· ${item}`)));
  root.append(capability);
  const redlines = el("div", "card");
  redlines.append(el("h3", "card-title", "可信边界（不可违反）"));
  [
    "p 值不是低产概率；预测区间不是保证范围。",
    "行动告警是需要人工复核的保守告警，不代表自动干预。",
    "空间证据只用于田块内相对定位，不是病因、病虫害或精确减产图。",
    "收获产量永不进入前瞻告警、报告或智能润色。",
    "原始一次性审计结果不可改写；修正口径是事后对照，不是第二次预注册审计。",
  ].forEach((item) => redlines.append(el("p", "", `· ${item}`)));
  root.append(redlines);
}

/* ================= 初始化与刷新 ================= */
async function refresh() {
  const [dashboard, alerts, reports, genLogs, spatial, knowledge, audit, retrospective] = await Promise.all([
    api("/api/v2/dashboard/summary"),
    api("/api/v2/alerts?limit=500"),
    api("/api/v2/reports?limit=400"),
    api("/api/v2/generation-logs"),
    api("/api/v2/fields/spatial-summary"),
    api("/api/v2/knowledge"),
    api("/api/v2/audit"),
    api("/api/v2/retrospective/summary"),
  ]);
  state.dashboard = dashboard;
  state.alerts = alerts.alerts;
  state.reports = reports;
  state.genLogs = genLogs;
  state.spatial = spatial;
  state.knowledge = knowledge;
  state.audit = audit;
  state.retrospective = retrospective;
  $("modeBadge").textContent = "前瞻演示 · 冻结审计数据";
  $("userChip").textContent = "本地演示 · 农艺师视角";
  const badgeNode = $("alertBadge");
  badgeNode.textContent = String(dashboard.portfolio.action_alerts);
  badgeNode.classList.toggle("hidden", dashboard.portfolio.action_alerts === 0);
  renderCurrentView();
}

function renderCurrentView() {
  if (state.view === "dashboard") renderDashboard();
  if (state.view === "alerts") renderAlerts();
  if (state.view === "reports") renderReports();
  if (state.view === "fields") renderFields();
  if (state.view === "value") renderValue();
  if (state.view === "tasks") renderTasks();
  if (state.view === "knowledge") renderKnowledge();
  if (state.view === "audit") renderAudit();
  if (state.view === "retrospective") renderRetrospective();
  if (state.view === "settings") renderSettings();
}

function switchView(view) {
  state.view = view;
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  renderCurrentView();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* ================= 演示导览 ================= */
const TOUR_STEPS = [
  {
    title: "第 1 步 · 总览：先看证据，再看结论",
    description: "322 个真实田块、12 个行动告警、23 个关注复查。权威审计结果如实展示——包括一次性冻结审计中未达标的分组。诚信本身就是产品力。",
    view: "dashboard",
  },
  {
    title: "第 2 步 · 两级告警设计",
    description: "关注复查 = 低成本巡田信号；行动告警 = 连续两期越界、需要农艺师复核的保守告警。按等级、国家、作物快速筛选，直接对应巡田资源怎么排。",
    view: "alerts",
  },
  {
    title: "第 3 步 · 预警证据：三个观测期",
    description: "打开一个行动告警田块，看三个观测期的预测产量、p 值与显著性水平的关系。每个数字都能追溯到冻结实验文件，p 值不等于低产概率。",
    view: "alerts",
    fieldId: "Argentina_DUP1_farm45_field633_soybean_2023",
    tab: "evidence",
  },
  {
    title: "第 4 步 · 专业中文报告",
    description: "规则引擎生成完整报告，可选用智能润色优化叙述（失败自动回退）；所有数字、风险等级、行动清单由规则锁定，人工批准后发布，支持一键打印。",
    view: "alerts",
    fieldId: "Argentina_DUP1_farm45_field633_soybean_2023",
    tab: "report",
  },
  {
    title: "第 5 步 · 现场闭环",
    description: "提交不可变现场快照（土壤、积水、症状、照片编号），进入任务与时间线；现场信息影响下一版报告，但永远不能修改冻结模型结论。",
    view: "alerts",
    fieldId: "Argentina_DUP1_farm45_field633_soybean_2023",
    tab: "field",
  },
  {
    title: "第 6 步 · 价值评估",
    description: "输入客户的田块数与巡田成本，现场计算预警后重点巡田能避免多少支出；再看 30 天试点路径与六个付费理由。",
    view: "value",
  },
];

let tourIndex = 0;

function openTour() {
  tourIndex = 0;
  $("tourDialog").showModal();
  renderTourStep();
}

function renderTourStep() {
  const step = TOUR_STEPS[tourIndex];
  $("tourStepLabel").textContent = `第 ${tourIndex + 1} / ${TOUR_STEPS.length} 步`;
  $("tourTitle").textContent = step.title;
  $("tourDescription").textContent = step.description;
  $("tourBackButton").disabled = tourIndex === 0;
  $("tourNextButton").style.display = tourIndex === TOUR_STEPS.length - 1 ? "none" : "";
  $("tourNextButton").textContent = tourIndex === TOUR_STEPS.length - 1 ? "" : "下一步";
  $("tourGoButton").textContent = tourIndex === TOUR_STEPS.length - 1 ? "完成" : "前往查看";
}

async function tourGo() {
  if (tourIndex === TOUR_STEPS.length - 1) {
    $("tourDialog").close();
    return;
  }
  const step = TOUR_STEPS[tourIndex];
  $("tourDialog").close();
  switchView(step.view);
  if (step.fieldId) {
    await openAlert(step.fieldId);
    if (step.tab) {
      state.detailTab = step.tab;
      renderDrawer();
    }
  }
}

$("demoTourButton").addEventListener("click", openTour);
$("closeTourButton").addEventListener("click", () => $("tourDialog").close());
$("tourBackButton").addEventListener("click", () => {
  tourIndex = Math.max(0, tourIndex - 1);
  renderTourStep();
});
$("tourNextButton").addEventListener("click", () => {
  tourIndex = Math.min(TOUR_STEPS.length - 1, tourIndex + 1);
  renderTourStep();
});
$("tourGoButton").addEventListener("click", tourGo);

document.querySelectorAll(".nav-item").forEach((button) => {
  button.addEventListener("click", () => switchView(button.dataset.view));
});
$("refreshButton").addEventListener("click", () => refresh().catch((error) => toast(error.message, true)));
$("drawerBackdrop").addEventListener("click", hideDrawer);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") hideDrawer();
});

refresh().catch((error) => toast(error.message, true));
