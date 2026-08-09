"""PhysCrop-Risk deterministic report v3, validator, and LLM slot builder.

Keeps the v2 envelope (farmer/professional/audit, actions, checklist,
provenance, review) so the existing LLM merge and workflow machinery can be
reused, while the semantics are the YieldSAT low-yield risk task.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from physcrop_agent_v2_core import (
    AGENT_ROOT,
    audit,
    canonical_hash,
    file_hash,
    init_db,
    load_knowledge,
    read_json,
    ulid,
    utc_now,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = AGENT_ROOT / "risk_evidence"
REPORT_DIR = AGENT_ROOT / "risk_reports"
KNOWLEDGE_PATH = AGENT_ROOT / "knowledge" / "agronomy_knowledge_risk.json"
REPORT_SCHEMA_VERSION = "physcrop_event_report_v3"
EVIDENCE_SCHEMA_VERSION = "physcrop_risk_alert_evidence_v1"

RISK_LABELS = {
    "action": "行动告警",
    "watch": "关注复查",
    "normal": "常规监测",
}

FORBIDDEN_PATTERNS = [
    r"低产概率(为|是|约|达|高达|等于)",
    r"(?<!不)(?<!未)(?<!非)(代表|意味着|表示|等同于)低产概率",
    r"已确诊|确诊为|模型确诊|可以确诊|能够确诊",
    r"亩用.{0,10}(克|毫升|公斤)",
    r"喷施.{0,20}(倍液|剂量)",
    r"由.{0,8}(缺水|涝害|病害|虫害|缺肥).{0,6}(造成|导致)",
    r"可以确定.{0,8}(缺水|涝害|病害|虫害|缺肥)",
    r"明确属于.{0,6}(涝害|病害|虫害|缺水)",
]

SAFE_PREFIXES = ("不会", "不能", "不得", "不要", "禁止", "不应", "不宜", "避免", "无需", "不进行", "未进行", "并非", "不是")
CONTEXTUAL_PHRASES = ("自动灌溉", "自动施肥", "自动施药", "自动喷药", "保证增产", "保证增收", "保证产量")


def _contains_forbidden_phrase(text: str, phrase: str) -> bool:
    for match in re.finditer(re.escape(phrase), text):
        prefix = text[max(0, match.start() - 5):match.start()]
        if any(safe in prefix for safe in SAFE_PREFIXES):
            continue
        return True
    return False


def load_risk_knowledge() -> dict[str, Any]:
    knowledge = read_json(KNOWLEDGE_PATH)
    if knowledge.get("review_status") != "approved_for_internal_pilot":
        raise ValueError("Risk knowledge base is not approved for use")
    return knowledge


def retrieve_risk_knowledge(evidence: dict[str, Any], knowledge: dict[str, Any]) -> list[dict[str, Any]]:
    country = evidence["identity"]["country"]
    crop = evidence["identity"]["crop"]
    source_index = {item["source_id"]: item for item in knowledge["sources"]}
    candidates = []
    for entry in knowledge["entries"]:
        countries = entry.get("countries") or []
        crops = entry.get("crops") or []
        if countries and country not in countries:
            continue
        if crops and crop not in crops:
            continue
        expanded = dict(entry)
        expanded["sources"] = [source_index[sid] for sid in entry["source_ids"]]
        expanded["specificity"] = len(countries) + len(crops)
        candidates.append(expanded)
    candidates.sort(key=lambda item: (-item["specificity"], item["knowledge_id"]))
    return candidates


def _allowed_action_inventory(entries: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    low = {action for entry in entries for action in entry["low_risk_actions"]}
    conditional = {action for entry in entries for action in entry["conditional_actions"]}
    return low, conditional


def _trajectory_assessment(evidence: dict[str, Any]) -> dict[str, Any]:
    history = [entry for entry in evidence["look_history"] if not entry.get("missing")]
    watches = [entry.get("watch_clean", 0) for entry in history]
    longest = current = 0
    for value in watches:
        current = current + 1 if value else 0
        longest = max(longest, current)
    first_watch = next((entry for entry in history if entry.get("watch_clean")), None)
    model = evidence["model_evidence"]
    boundary = evidence["current_information_boundary"]
    narrative = (
        f"该田块在 {len(history)} 个可用观测期中有 {sum(watches)} 个观测期越过校准阈值，"
        f"最长连续 {longest} 个观测期。当前观测期为第 {boundary['look']} 期，预测产量 "
        f"{model['prediction_t_ha']:.2f} 吨/公顷，p 值 {model['null_pvalue']:.4f} 与 "
        f"观测期显著性水平 {model['look_alpha']:.3f} 比较用于决定关注或行动告警，不代表低产概率。"
    )
    return {
        "observation_count": len(history),
        "above_threshold_count": sum(watches),
        "longest_consecutive_above": longest,
        "first_above_threshold_date": first_watch.get("cutoff_date") if first_watch else None,
        "latest_look": boundary["look"],
        "latest_prediction_t_ha": model["prediction_t_ha"],
        "latest_pvalue": model["null_pvalue"],
        "latest_alpha": model["look_alpha"],
        "watch_original": model["watch_original"],
        "action_original": model["action_original"],
        "watch_clean": model["watch"],
        "action_clean": model["action"],
        "narrative": narrative,
        "source_paths": ["look_history", "model_evidence", "current_information_boundary"],
    }


def _spatial_assessment(evidence: dict[str, Any]) -> dict[str, Any]:
    spatial = evidence["spatial_evidence"]
    if not spatial.get("available"):
        return {
            "available": False,
            "narrative": "当前田块没有可用的空间残差图；空间证据不作为本报告的一部分。",
            "interpretation_boundary": spatial["interpretation_boundary"],
            "source_paths": ["spatial_evidence"],
        }
    narrative = (
        f"空间证据可用：{spatial['pixels']} 个像素，田块内残差均方根误差 "
        f"{spatial['rmse_spatial_t_ha']:.3f} 吨/公顷，田块内秩相关 {spatial['within_field_rho']:.3f}，"
        f"底部两成区域定位能力 AUC {spatial['bottom20_auroc']:.3f}。"
        "该图只用于定位田块内相对风险区域，不是病因、病虫害或精确减产图。"
    )
    return {
        "available": True,
        "narrative": narrative,
        "metrics": {
            "pixels": spatial["pixels"],
            "rmse_spatial_t_ha": spatial["rmse_spatial_t_ha"],
            "r2_spatial": spatial["r2_spatial"],
            "within_field_rho": spatial["within_field_rho"],
            "bottom20_auroc": spatial["bottom20_auroc"],
        },
        "interpretation_boundary": spatial["interpretation_boundary"],
        "source_paths": ["spatial_evidence"],
    }


def _evidence_items(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    model = evidence["model_evidence"]
    uncertainty = evidence["uncertainty"]
    return [
        {
            "title": "预测产量",
            "field_path": "model_evidence.prediction_t_ha",
            "value": model["prediction_t_ha"],
            "interpretation": "当前 look 可用观测下的最终产量点估计。",
            "limitation": "点估计不是保证范围，也不是低产概率。",
        },
        {
            "title": "训练定义低产 cutoff",
            "field_path": "model_evidence.training_defined_low_yield_cutoff_t_ha",
            "value": model["training_defined_low_yield_cutoff_t_ha"],
            "interpretation": "由模型拟合分区 q20 与预声明层级确定的低产阈值。",
            "limitation": "阈值来自训练分布，不代表经营目标或保险条款。",
        },
        {
            "title": "越界检验 p 值",
            "field_path": "model_evidence.null_pvalue",
            "value": model["null_pvalue"],
            "interpretation": "当前分数相对非低产参考分布的越界程度。",
            "limitation": "p 值不是低产概率。",
        },
        {
            "title": "观测期显著性水平",
            "field_path": "model_evidence.look_alpha",
            "value": model["look_alpha"],
            "interpretation": "本观测期分配的族系错误预算份额。",
            "limitation": "三个观测期之和为 0.05，单期越界不等于确定风险。",
        },
        {
            "title": "关注信号",
            "field_path": "model_evidence.watch",
            "value": model["watch"],
            "interpretation": "当前决策口径下是否有单观测期越界。",
            "limitation": "关注信号表示需要复查，不表示确诊。",
        },
        {
            "title": "行动告警",
            "field_path": "model_evidence.action",
            "value": model["action"],
            "interpretation": "当前决策口径下是否连续两个观测期越界。",
            "limitation": "行动告警是需要人工复核的保守告警，不表示自动干预。",
        },
        {
            "title": "90% 上界",
            "field_path": "uncertainty.upper_90",
            "value": uncertainty["upper_90"],
            "interpretation": "单侧 split-conformal 90% 上界。",
            "limitation": "上界是预测证据，不是保证范围。",
        },
        {
            "title": "95% 上界",
            "field_path": "uncertainty.upper_95",
            "value": uncertainty["upper_95"],
            "interpretation": "单侧 split-conformal 95% 上界。",
            "limitation": "上界是预测证据，不是保证范围。",
        },
    ]


def deterministic_risk_report(evidence: dict[str, Any], knowledge: dict[str, Any]) -> dict[str, Any]:
    identity = evidence["identity"]
    model = evidence["model_evidence"]
    boundary = evidence["current_information_boundary"]
    entries = retrieve_risk_knowledge(evidence, knowledge)
    low_actions, conditional_actions = _allowed_action_inventory(entries)
    level = "action" if model["action"] else ("watch" if model["watch"] else "normal")
    label = RISK_LABELS[level]
    ordered_low = sorted(low_actions)
    if level == "action":
        selected_low = ordered_low[:5]
        selected_conditional = sorted(conditional_actions)
    elif level == "watch":
        selected_low = ordered_low[:4]
        selected_conditional = []
    else:
        selected_low = ordered_low[:3]
        selected_conditional = []
    trajectory = _trajectory_assessment(evidence)
    spatial = _spatial_assessment(evidence)
    status_text = (
        f"连续两个 look 越界形成行动告警" if model["action"]
        else f"当前 look 越界形成关注复查" if model["watch"]
        else "当前尚未形成 watch 或 action"
    )
    headline = f"{label}：{status_text}，需结合现场信息判断原因。"
    candidate_issues = [
        {
            "knowledge_id": entry["knowledge_id"],
            "title": entry["title"],
            "status": "待现场排查",
            "explanation": entry["candidate_issue"],
            "confidence": "证据有限",
            "supporting_evidence": ["低产风险信号", "当前观测期的预测与 p 值"],
            "missing_information": [
                "现场症状照片与对照区记录",
                "土壤水分、积排水和近期农事记录",
                "与 look 日期对齐的天气观测",
            ],
            "field_questions": [
                "风险区与对照区是否存在可重复的植株或冠层差异？",
                "土壤水分、积排水和灌排设施状态如何？",
                "近期降水、农事操作与风险出现时间是否吻合？",
            ],
        }
        for entry in entries
    ]
    citations = []
    seen_sources: set[str] = set()
    for entry in entries:
        for source in entry["sources"]:
            if source["source_id"] not in seen_sources:
                citations.append(source)
                seen_sources.add(source["source_id"])
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_id": f"RPT-{identity['field_id']}-{ulid()}",
        "event_key": identity["field_id"],
        "language": "zh-CN",
        "status": "needs_review",
        "generated_at": utc_now(),
        "risk": {
            "level": level,
            "label": label,
            "decision_source": "deterministic_rule",
            "not_a_diagnosis": True,
            "policy": "post_audit_clean_null",
        },
        "farmer_summary": {
            "headline": headline,
            "what_we_observed": (
                f"{identity['country']} {identity['crop']} {identity['season']} 季节第 "
                f"{boundary['look']} 个观测时点预测产量为 {model['prediction_t_ha']:.2f} 吨/公顷，"
                f"训练定义低产阈值为 {model['training_defined_low_yield_cutoff_t_ha']:.2f} 吨/公顷。"
            ),
            "what_to_do_now": selected_low[:4],
            "important_note": "模型只能提示低产风险，不能仅凭遥感确定缺水、病虫害、缺肥或产量损失。",
        },
        "professional_analysis": {
            "conclusion": headline,
            "narrative": (
                f"当前风险等级为{label}。预测产量 {model['prediction_t_ha']:.2f} 吨/公顷，"
                f"低于训练定义低产阈值 {model['training_defined_low_yield_cutoff_t_ha']:.2f} 吨/公顷 "
                f"的差距由风险分表达；p 值 {model['null_pvalue']:.4f} 与 "
                f"观测期显著性水平 {model['look_alpha']:.3f} 比较决定关注或行动告警。"
                "p 值不是低产概率，预测区间也不是保证范围；候选原因仍需现场排查。"
            ),
            "event_overview": {
                "country": identity["country"],
                "crop": identity["crop"],
                "season": identity["season"],
                "field_id": identity["field_id"],
                "farm_id": identity["farm_id"],
                "current_look": boundary["look"],
                "look_type": boundary["look_type"],
                "gdd_cutoff": boundary["gdd_cutoff"],
                "cutoff_date": boundary["cutoff_date"],
                "analysis_mode": evidence["mode"],
                "decision_policy": model["decision_policy"],
            },
            "trajectory_assessment": trajectory,
            "spatial_assessment": spatial,
            "evidence_items": _evidence_items(evidence),
            "candidate_issues": candidate_issues,
            "data_quality": evidence["data_quality"],
        },
        "actions": {
            "low_risk_now": selected_low,
            "after_field_confirmation": selected_conditional,
            "automatically_executed": [],
        },
        "action_plan": {
            "rationale": (
                f"当前为{label}，处置目标是先用低风险现场检查补齐信息，再由人工决定是否升级。"
                "系统不会自动触发灌溉、施肥、施药或植保操作。"
            ),
            "within_24_hours": selected_low[:4],
            "within_24_to_72_hours": [
                action for action in selected_low
                if any(token in action for token in ("下一次", "24 至 72", "复查"))
            ],
            "after_confirmation": selected_conditional,
            "do_not_do": [
                "不要把 p 值解释为低产概率或保证范围。",
                "不要在缺少现场确认时直接施药、追肥或改变灌溉计划。",
                "不要把空间证据解释为病因、病虫害或精确减产图。",
            ],
        },
        "field_checklist": [
            {
                "item": "确认作物、品种和生育期", "required": True,
                "method": "核对种植档案或向地块负责人确认。",
                "record": "作物、品种、生育期、播种日期。",
            },
            {
                "item": "风险区与对照区对照拍照", "required": True,
                "method": "中心、边缘、邻近正常区采用相近视角各拍摄至少一张。",
                "record": "照片编号、时间、位置和拍摄方向。",
            },
            {
                "item": "记录土壤水分、积水和灌排状态", "required": True,
                "method": "在风险区和对照区分别检查表层及根层，并检查灌排设施。",
                "record": "点位、土壤水分等级、积水范围、灌排异常。",
            },
            {
                "item": "记录冠层和植株可见症状", "required": True,
                "method": "观察叶片、茎秆、倒伏、缺苗、斑点和虫体，避免先入为主判断病因。",
                "record": "症状、发生比例、空间边界和对照差异。",
            },
        ],
        "recheck_plan": {
            "recommended_window": "下一次有效卫星 look 或 24 至 72 小时内",
            "compare": [
                "预测与低产阈值的差距是否收窄",
                "p 值与观测期显著性水平的关系是否持续越界",
                "现场症状是否持续或加重",
            ],
            "close_condition": "后续观测期未越界、现场未见异常且数据质量问题得到解释，可降级为常规监测。",
            "escalate_condition": "形成 action、风险范围扩大或现场发现明确症状时，升级给农艺师。",
        },
        "escalation": {
            "rule": "连续两个 look 越界形成 action，或现场出现明确症状且复查无法解释时，交由农艺师复核。",
            "human_approval_required": True,
        },
        "knowledge_citations": citations,
        "limitations": evidence["data_quality"]["limitations"] + [
            "本证据来自冻结审计产物重建；真实部署需要前瞻运行验证。",
        ],
        "provenance": {
            "evidence_hash": evidence["evidence_hash"],
            "knowledge_version": knowledge["version"],
            "generator": "deterministic_fallback",
            "model": "PhysCrop-Risk dual-tree",
            "adapter_version": evidence["provenance"]["adapter_version"],
        },
        "review": {"decision": "pending", "reviewer": None, "comment": None},
    }
    return report


def _get_path(value: dict[str, Any], path: str) -> Any:
    current = value
    for segment in path.split("."):
        current = current[segment]
    return current


def validate_risk_report(report: dict[str, Any], evidence: dict[str, Any],
                         knowledge: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    required = {
        "schema_version", "event_key", "language", "status", "risk", "farmer_summary",
        "professional_analysis", "actions", "action_plan", "field_checklist", "recheck_plan",
        "escalation", "knowledge_citations", "limitations", "provenance", "review",
    }
    missing = sorted(required - set(report))
    if missing:
        errors.append(f"缺少必需字段: {', '.join(missing)}")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        errors.append("报告 schema_version 不匹配")
    if report.get("event_key") != evidence["identity"]["field_id"]:
        errors.append("event_key 与证据不一致")
    if report.get("language") != "zh-CN":
        errors.append("报告必须输出 zh-CN")
    model = evidence["model_evidence"]
    expected_level = "action" if model["action"] else ("watch" if model["watch"] else "normal")
    if report.get("risk", {}).get("level") != expected_level:
        errors.append("LLM 修改了规则确定的风险等级")
    if report.get("risk", {}).get("decision_source") != "deterministic_rule":
        errors.append("风险等级必须标记为 deterministic_rule")
    if report.get("actions", {}).get("automatically_executed"):
        errors.append("系统禁止自动执行农业处置")
    entries = retrieve_risk_knowledge(evidence, knowledge)
    allowed_low, allowed_conditional = _allowed_action_inventory(entries)
    supplied_low = set(report.get("actions", {}).get("low_risk_now", []))
    supplied_conditional = set(report.get("actions", {}).get("after_field_confirmation", []))
    if not supplied_low <= allowed_low:
        errors.append(f"包含知识库未批准的即时行动: {sorted(supplied_low - allowed_low)}")
    if not supplied_conditional <= allowed_conditional:
        errors.append(f"包含知识库未批准的条件行动: {sorted(supplied_conditional - allowed_conditional)}")
    action_plan = report.get("action_plan", {})
    planned_low = set(action_plan.get("within_24_hours", [])) | set(action_plan.get("within_24_to_72_hours", []))
    if not planned_low <= supplied_low:
        errors.append("分时段行动计划包含即时行动清单之外的内容")
    if set(action_plan.get("after_confirmation", [])) != supplied_conditional:
        errors.append("确认后行动计划与知识库批准清单不一致")
    if expected_level in {"normal", "watch"} and action_plan.get("after_confirmation"):
        errors.append("非 action 事件不应包含确认后处置")
    allowed_sources = {source["source_id"] for entry in entries for source in entry["sources"]}
    used_sources = {item.get("source_id") for item in report.get("knowledge_citations", [])}
    if not used_sources <= allowed_sources:
        errors.append(f"引用了未检索到的知识来源: {sorted(used_sources - allowed_sources)}")
    for item in report.get("professional_analysis", {}).get("evidence_items", []):
        path = item.get("field_path", "")
        try:
            expected = _get_path(evidence, path)
        except (KeyError, TypeError):
            errors.append(f"无效证据路径: {path}")
            continue
        if item.get("value") != expected:
            errors.append(f"证据值与事实源不一致: {path}")
    narrative = report.get("professional_analysis", {}).get("narrative", "")
    required_terms = [
        RISK_LABELS[expected_level],
        f"{model['prediction_t_ha']:.2f}",
        f"{model['null_pvalue']:.4f}",
    ]
    if any(term not in narrative for term in required_terms):
        errors.append("专业叙事缺少风险等级、预测或 p-value")
    if "不是低产概率" not in narrative and ("不是" not in narrative or "概率" not in narrative):
        errors.append("专业叙事没有说明 p-value 不是概率")
    if "现场" not in narrative or not any(term in narrative for term in ("排查", "核验")):
        errors.append("专业叙事缺少现场排查边界")
    trajectory = report.get("professional_analysis", {}).get("trajectory_assessment", {})
    expected_trajectory = _trajectory_assessment(evidence)
    for key, expected in expected_trajectory.items():
        if key != "narrative" and trajectory.get(key) != expected:
            errors.append(f"轨迹统计与证据不一致: {key}")
    if "不代表低产概率" not in trajectory.get("narrative", ""):
        errors.append("轨迹叙事缺少 p-value 解释边界")
    spatial = report.get("professional_analysis", {}).get("spatial_assessment", {})
    if evidence["spatial_evidence"]["available"]:
        if spatial.get("available") is not True:
            errors.append("空间证据可用性被修改")
        if "不是病因" not in spatial.get("narrative", ""):
            errors.append("空间叙事缺少非诊断边界")
    if evidence.get("mode") == "prospective_inference" and "retrospective_block" in evidence:
        errors.append("前瞻证据不应包含回顾块")
    content = json.dumps(report, ensure_ascii=False)
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, content):
            errors.append(f"命中高风险或越界表达: {pattern}")
    for phrase in CONTEXTUAL_PHRASES:
        if _contains_forbidden_phrase(content, phrase):
            errors.append(f"命中高风险或越界表达: {phrase}")
    if not report.get("escalation", {}).get("human_approval_required"):
        errors.append("缺少人工批准门槛")
    if not report.get("risk", {}).get("not_a_diagnosis"):
        errors.append("缺少非诊断声明")
    return {"passes": not errors, "errors": errors, "warnings": warnings}


def build_risk_llm_messages(evidence: dict[str, Any], knowledge: dict[str, Any],
                            draft: dict[str, Any]) -> list[dict[str, str]]:
    entries = retrieve_risk_knowledge(evidence, knowledge)
    compact_evidence = {
        "identity": evidence["identity"],
        "current_information_boundary": evidence["current_information_boundary"],
        "model_evidence": evidence["model_evidence"],
        "uncertainty": evidence["uncertainty"],
        "spatial_evidence": evidence["spatial_evidence"],
        "data_quality": evidence["data_quality"],
        "evidence_hash": evidence["evidence_hash"],
    }
    system = (
        "你是 PhysCrop-Risk 低产风险解释 Agent。你只能重写用户提供的 JSON 草稿中的中文解释，"
        "必须返回单个合法 JSON 对象。硬约束："
        "1. 不得修改 event_key、风险等级、任何数值、日期、证据路径、行动清单、引用、溯源和审核状态。"
        "2. p-value 是越界证据，不是低产概率；预测区间不是保证范围。"
        "3. 不得诊断具体病虫害、缺水、涝害、缺素或产量损失；候选原因必须写成待排查。"
        "4. 不得新增农药、肥料、灌溉剂量或任何知识库外行动。"
        "5. 用清楚、克制、可执行的简体中文；区分农户摘要和专业分析。"
        "6. action 是需要人工复核的保守告警，不代表自动干预。"
        "7. 空间证据只能用于田块内相对定位，不能解释为病因图。"
        "8. 锁定数字、日期、p-value、alpha、look 与字段名必须原样保留。"
    )
    user = {
        "task": (
            "不要复制 draft_report。返回一个精简 JSON，只包含 farmer_summary.headline、"
            "farmer_summary.important_note、professional_analysis.conclusion，以及四个新增字段："
            "professional_analysis.narrative_addition、professional_analysis.trajectory_assessment.narrative_addition、"
            "professional_analysis.spatial_assessment.narrative_addition、action_plan.rationale_addition。"
            "每个 addition 写 2 至 3 句、60 至 180 字的补充解释，且不得出现任何阿拉伯数字。"
        ),
        "evidence": compact_evidence,
        "retrieved_knowledge": entries,
        "draft_report": draft,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def _persist_risk_report(connection, report: dict[str, Any], validation: dict[str, Any],
                         generator: str, provider: str = "local", model: str = "rules-v3") -> Path:
    if not validation["passes"]:
        raise ValueError(f"Refusing to persist invalid report: {validation['errors']}")
    report = json.loads(json.dumps(report, ensure_ascii=False))
    report["status"] = "needs_review"
    report["provenance"]["generator"] = generator
    path = REPORT_DIR / f"{report['event_key']}.json"
    write_json(path, report)
    now = utc_now()
    connection.execute(
        """INSERT INTO reports
           (report_id,event_key,status,generator,provider,model,report_json,evidence_hash,
            knowledge_version,validation_json,created_at,updated_at,
            payload_uri,payload_sha256,report_version,parent_report_id,superseded_by)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            report["report_id"], report["event_key"], "needs_review", generator, provider, model,
            json.dumps(report, ensure_ascii=False), report["provenance"]["evidence_hash"],
            report["provenance"]["knowledge_version"], json.dumps(validation, ensure_ascii=False),
            report["generated_at"], now, f"risk_reports/{path.name}", file_hash(path),
            "1", "", "",
        ),
    )
    connection.commit()
    return path


persist_risk_report = _persist_risk_report


def generate_risk_deterministic_reports() -> list[Path]:
    knowledge = load_risk_knowledge()
    connection = init_db()
    paths = []
    try:
        for evidence_path in sorted(EVIDENCE_DIR.glob("*.json")):
            if evidence_path.stem == "risk_manifest":
                continue
            evidence = read_json(evidence_path)
            report = deterministic_risk_report(evidence, knowledge)
            validation = validate_risk_report(report, evidence, knowledge)
            paths.append(_persist_risk_report(connection, report, validation, "deterministic_fallback"))
        audit(connection, "system", "generate_risk_reports", "all_risk_fields", {"count": len(paths)})
    finally:
        connection.close()
    return paths


def main() -> None:
    paths = generate_risk_deterministic_reports()
    print(json.dumps({"report_count": len(paths), "report_dir": str(REPORT_DIR)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
