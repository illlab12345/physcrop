"""Local/internal-pilot web service for the PhysCrop event response agent.

API credentials remain server-side. Use --demo only on localhost; production
deployments must provide PHYSCROP_AGENT_USERS_JSON and terminate TLS upstream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

from physcrop_agent_v2_core import (
    AGENT_ROOT,
    CONFIRMATION,
    DB_PATH,
    EVIDENCE_DIR,
    LOG_DIR,
    REPORT_DIR,
    audit,
    build_llm_messages,
    deterministic_report,
    init_db,
    load_event,
    load_knowledge,
    load_report,
    merge_llm_narrative,
    parse_llm_json,
    persist_report,
    read_json,
    read_csv,
    ulid,
    utc_now,
    validate_report,
    write_json,
)
from physcrop_agent_v2_risk_report import (
    REPORT_DIR as RISK_REPORT_DIR,
    build_risk_llm_messages,
    deterministic_risk_report,
    load_risk_knowledge,
    persist_risk_report,
    validate_risk_report,
)
from physcrop_report_pdf import build_report_pdf


ROLE_LEVEL = {"farmer": 1, "agronomist": 2, "admin": 3}
DEMO_CONFIG_PATH = AGENT_ROOT / "commercial_demo_config.json"
MAIN_TABLE_PATH = CONFIRMATION / "wacv_tables_v5" / "wacv_v5_main_table.csv"
RISK_EVIDENCE_DIR = AGENT_ROOT / "risk_evidence"
RISK_OBSERVATION_DIR = AGENT_ROOT / "risk_observations"
GENERATION_LOG_DIR = AGENT_ROOT / "risk_reports" / "logs"
LEGACY_APP_DIR = AGENT_ROOT / "app"
V2_APP_DIR = AGENT_ROOT / "app_v3"
LEGACY_SOURCE_TYPE = "anda_controlled_v2"
RISK_SOURCE_TYPE = "yieldsat_risk_v1"
PROVIDER_ZH_SERVER = {"local": "规则引擎", "deepseek": "深度求索", "openai": "OpenAI", "anthropic": "Anthropic"}
WORKFLOW_LABELS = {
    "new": "新告警",
    "assigned": "待现场核验",
    "in_field": "巡田中",
    "feedback_received": "已反馈",
    "under_review": "农艺审核",
    "closed": "已关闭",
}
ALLOWED_TRANSITIONS = {
    "new": {"assigned"},
    "assigned": {"in_field", "new"},
    "in_field": {"feedback_received", "assigned"},
    "feedback_received": {"under_review", "in_field"},
    "under_review": {"closed", "feedback_received"},
    "closed": {"under_review"},
}

_JOBS: dict[str, dict[str, Any]] = {}
_EXECUTOR = ThreadPoolExecutor(max_workers=2)


def _add_job_event(job_id: str, stage: str, message: str) -> None:
    job = _JOBS.get(job_id)
    if job is None:
        return
    job["stage"] = stage
    job.setdefault("events", []).append({"stage": stage, "message": message, "at": utc_now()})


def configured_users() -> dict[str, dict[str, str]]:
    raw = os.environ.get("PHYSCROP_AGENT_USERS_JSON", "")
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("PHYSCROP_AGENT_USERS_JSON must be a token-to-user object")
    users = {}
    for token, user in value.items():
        if not isinstance(user, dict) or user.get("role") not in ROLE_LEVEL:
            raise ValueError("Each configured user needs username and farmer/agronomist/admin role")
        users[str(token)] = {
            "username": str(user.get("username", "user")),
            "role": user["role"],
            "farms": [str(item) for item in (user.get("farms") or [])],
        }
    return users


def provider_config(provider: str) -> tuple[str, str, dict[str, str]]:
    if provider == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        return (
            os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions"),
            os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            {"Authorization": f"Bearer {key}"},
        )
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        return (
            os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"),
            os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
            {"Authorization": f"Bearer {key}"},
        )
    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        return (
            os.environ.get("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages"),
            os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
    raise ValueError(f"Unsupported provider: {provider}")


def call_llm(provider: str, messages: list[dict[str, str]]) -> tuple[dict[str, Any], str]:
    endpoint, model, headers = provider_config(provider)
    has_key = (
        bool(headers.get("x-api-key")) if provider == "anthropic"
        else headers.get("Authorization", "") != "Bearer "
    )
    if not has_key:
        raise RuntimeError(f"{provider} API key is not configured on the server")
    if provider == "anthropic":
        system = next(item["content"] for item in messages if item["role"] == "system")
        body = {
            "model": model,
            "max_tokens": 5000,
            "temperature": 0,
            "system": system,
            "messages": [item for item in messages if item["role"] != "system"],
        }
    else:
        body = {
            "model": model,
            "temperature": 0,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"{provider} API returned HTTP {exc.code}: {detail}") from exc
    if provider == "anthropic":
        content = "".join(item.get("text", "") for item in payload.get("content", []))
    else:
        content = payload["choices"][0]["message"]["content"]
    return parse_llm_json(content), model


class AgentHandler(SimpleHTTPRequestHandler):
    server_version = "PhysCropAgent/2.0"

    def __init__(self, *args: Any, directory: str, demo: bool, users: dict[str, dict[str, str]], **kwargs: Any):
        self.demo = demo
        self.users = users
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'",
        )
        super().end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _json(self, value: Any, status: int = 200) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message, "status": status}, status)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 2_000_000:
            raise ValueError("Request body too large")
        value = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def _user(self, minimum: str = "farmer") -> dict[str, str] | None:
        if self.demo:
            user = {"username": "本地演示用户", "role": "agronomist"}
        else:
            auth = self.headers.get("Authorization", "")
            token = auth[7:] if auth.startswith("Bearer ") else ""
            user = self.users.get(token)
        if user is None:
            self._error(HTTPStatus.UNAUTHORIZED, "需要有效的访问令牌")
            return None
        if ROLE_LEVEL[user["role"]] < ROLE_LEVEL[minimum]:
            self._error(HTTPStatus.FORBIDDEN, "当前角色无权执行该操作")
            return None
        return user

    def _connection(self) -> sqlite3.Connection:
        connection = init_db()
        connection.row_factory = sqlite3.Row
        return connection

    # ------------------------------------------------------------------
    # API v2: PhysCrop-Risk workspace
    # ------------------------------------------------------------------
    def _v2_user(self, minimum: str = "farmer") -> dict[str, Any] | None:
        user = self._user(minimum)
        if user is None:
            return None
        if self.demo:
            user["farms"] = []
        else:
            token = self.headers.get("Authorization", "")[7:]
            configured = self.users.get(token, {})
            user["farms"] = configured.get("farms") or []
        return user

    def _v2_field_allowed(self, user: dict[str, Any], field_id: str) -> bool:
        farms = set(user.get("farms") or [])
        if not farms:
            return True
        connection = self._connection()
        try:
            row = connection.execute(
                "SELECT farm_id FROM fields WHERE field_id=?", (field_id,)
            ).fetchone()
        finally:
            connection.close()
        return bool(row and row["farm_id"] in farms)

    def _v2_risk_evidence(self, field_id: str) -> dict[str, Any]:
        path = RISK_EVIDENCE_DIR / f"{field_id}.json"
        if not path.exists():
            raise FileNotFoundError(field_id)
        return read_json(path)

    def _v2_alert_items(self, user: dict[str, Any]) -> list[dict[str, Any]]:
        connection = self._connection()
        farms = set(user.get("farms") or [])
        try:
            sql = (
                "SELECT e.event_key,e.risk_level,e.detected,e.country,e.crop,e.season_year,"
                "e.watch_original,e.action_original,e.watch_clean,e.action_clean,e.current_look,"
                "e.workflow_status,e.assignee,e.updated_at,f.farm_id "
                "FROM events e LEFT JOIN fields f ON f.field_id=e.event_key "
                "WHERE e.source_type=? "
            )
            params: list[Any] = [RISK_SOURCE_TYPE]
            if farms:
                sql += "AND f.farm_id IN (%s) " % ",".join("?" * len(farms))
                params.extend(sorted(farms))
            sql += "ORDER BY e.action_clean DESC, e.watch_clean DESC, e.updated_at DESC"
            rows = connection.execute(sql, params).fetchall()
        finally:
            connection.close()
        items = []
        for row in rows:
            level = "action" if row["action_clean"] else ("watch" if row["watch_clean"] else "normal")
            items.append({
                "event_key": row["event_key"],
                "field_id": row["event_key"],
                "country": row["country"],
                "crop": row["crop"],
                "season": str(row["season_year"]),
                "farm_id": row["farm_id"],
                "risk_level": level,
                "risk_label": {"action": "行动告警", "watch": "关注复查", "normal": "常规监测"}[level],
                "watch_clean": row["watch_clean"],
                "action_clean": row["action_clean"],
                "watch_original": row["watch_original"],
                "action_original": row["action_original"],
                "current_look": row["current_look"],
                "workflow_status": row["workflow_status"],
                "workflow_label": WORKFLOW_LABELS.get(row["workflow_status"], row["workflow_status"]),
                "assignee": row["assignee"],
                "updated_at": row["updated_at"],
            })
        return items

    def _v2_alerts(self, user: dict[str, Any]) -> None:
        query = parse_qs(urlparse(self.path).query)
        limit = min(int(query.get("limit", ["100"])[0]), 500)
        cursor = int(query.get("cursor", ["0"])[0])
        country = query.get("country", [""])[0]
        crop = query.get("crop", [""])[0]
        level = query.get("level", [""])[0]
        items = self._v2_alert_items(user)
        if country:
            items = [item for item in items if item["country"] == country]
        if crop:
            items = [item for item in items if item["crop"] == crop]
        if level:
            items = [item for item in items if item["risk_level"] == level]
        page = items[cursor:cursor + limit]
        self._json({"alerts": page, "count": len(items), "cursor": cursor + len(page),
                    "has_more": cursor + len(page) < len(items)})

    def _v2_alert_detail(self, user: dict[str, Any], field_id: str) -> None:
        if not self._v2_field_allowed(user, field_id):
            self._error(HTTPStatus.FORBIDDEN, "无权访问该田块")
            return
        evidence = self._v2_risk_evidence(field_id)
        connection = self._connection()
        try:
            event = connection.execute(
                "SELECT * FROM events WHERE event_key=? AND source_type=?", (field_id, RISK_SOURCE_TYPE)
            ).fetchone()
            if event is None:
                raise FileNotFoundError(field_id)
            report = connection.execute(
                "SELECT * FROM reports WHERE event_key=? ORDER BY updated_at DESC LIMIT 1", (field_id,)
            ).fetchone()
            report_payload = json.loads(report["report_json"]) if report else None
            snapshots = [dict(row) for row in connection.execute(
                "SELECT * FROM observation_snapshots WHERE event_key=? ORDER BY created_at DESC", (field_id,)
            ).fetchall()]
            observations = [dict(row) for row in connection.execute(
                "SELECT * FROM field_observations WHERE event_key=? ORDER BY created_at DESC", (field_id,)
            ).fetchall()]
            tasks = [dict(row) for row in connection.execute(
                "SELECT * FROM tasks WHERE event_key=? ORDER BY updated_at DESC", (field_id,)
            ).fetchall()]
            timeline = [dict(row) for row in connection.execute(
                "SELECT * FROM event_timeline WHERE event_key=? ORDER BY created_at DESC", (field_id,)
            ).fetchall()]
            report_versions = [dict(row) for row in connection.execute(
                """SELECT report_id,status,generator,provider,model,created_at
                   FROM reports WHERE event_key=? ORDER BY created_at DESC LIMIT 8""",
                (field_id,),
            ).fetchall()]
            def latest_payload(generator: str) -> dict[str, Any] | None:
                row = connection.execute(
                    "SELECT report_json FROM reports WHERE event_key=? AND generator=? "
                    "ORDER BY created_at DESC LIMIT 1",
                    (field_id, generator),
                ).fetchone()
                return json.loads(row["report_json"]) if row else None
            compare = {
                "rule": latest_payload("deterministic_fallback"),
                "llm": latest_payload("llm_guarded"),
            }
        finally:
            connection.close()
        self._json({
            "evidence": evidence,
            "report": report_payload,
            "compare": compare,
            "report_versions": report_versions,
            "snapshots": snapshots,
            "observations": observations,
            "workflow": {
                "status": event["workflow_status"],
                "label": WORKFLOW_LABELS.get(event["workflow_status"], event["workflow_status"]),
                "assignee": event["assignee"],
                "due_at": event["due_at"],
                "closed_reason": event["closed_reason"],
            },
            "tasks": tasks,
            "timeline": timeline,
        })

    def _v2_dashboard_summary(self) -> None:
        connection = self._connection()
        try:
            total = connection.execute(
                "SELECT COUNT(*) FROM events WHERE source_type=?", (RISK_SOURCE_TYPE,)
            ).fetchone()[0]
            action = connection.execute(
                "SELECT COUNT(*) FROM events WHERE source_type=? AND action_clean=1", (RISK_SOURCE_TYPE,)
            ).fetchone()[0]
            watch = connection.execute(
                "SELECT COUNT(*) FROM events WHERE source_type=? AND watch_clean=1 AND action_clean=0",
                (RISK_SOURCE_TYPE,),
            ).fetchone()[0]
            pending_review = connection.execute(
                """SELECT COUNT(*) FROM reports r JOIN events e ON e.event_key=r.event_key
                   WHERE e.source_type=? AND r.status='needs_review'""", (RISK_SOURCE_TYPE,)
            ).fetchone()[0]
            open_tasks = connection.execute(
                "SELECT COUNT(*) FROM tasks WHERE status!='completed'"
            ).fetchone()[0]
            snapshots = connection.execute("SELECT COUNT(*) FROM observation_snapshots").fetchone()[0]
            risk_counts = {row["risk_level"]: row["n"] for row in connection.execute(
                """SELECT CASE WHEN action_clean=1 THEN 'action' WHEN watch_clean=1 THEN 'watch' ELSE 'normal' END AS risk_level,
                          COUNT(*) AS n FROM events WHERE source_type=? GROUP BY risk_level""",
                (RISK_SOURCE_TYPE,),
            )}
            country_counts = {row["country"]: row["n"] for row in connection.execute(
                "SELECT country, COUNT(*) AS n FROM events WHERE source_type=? GROUP BY country",
                (RISK_SOURCE_TYPE,),
            )}
            crop_counts = {row["crop"]: row["n"] for row in connection.execute(
                "SELECT crop, COUNT(*) AS n FROM events WHERE source_type=? GROUP BY crop",
                (RISK_SOURCE_TYPE,),
            )}
        finally:
            connection.close()
        authoritative = {}
        final_report = read_json(Path(__file__).resolve().parents[1]
                                 / "results/yieldsat_risk_v1/final_audit/final_report.json")
        p0b_report = read_json(Path(__file__).resolve().parents[1]
                               / "results/yieldsat_risk_v1/post_audit_p0p1/p0b_report.json")
        authoritative = {
            "audit_status": final_report["status"],
            "audit_fields": final_report["audit_fields"],
            "look3_rmse": final_report["looks"]["3"]["metrics"]["rmse"],
            "look3_r2": final_report["looks"]["3"]["metrics"]["r2"],
            "look3_auroc": final_report["looks"]["3"]["metrics"]["auroc"],
            "clean_null_watch": p0b_report["audit_uniform_policy_comparison"]["dual_tree"]["watch"],
            "clean_null_action": p0b_report["audit_uniform_policy_comparison"]["dual_tree"]["action"],
            "disclosure": "冻结一次性审计不可改写；clean-null 为 post-audit 修正比较。",
        }
        self._json({
            "portfolio": {
                "total_fields": total,
                "action_alerts": action,
                "watch_alerts": watch,
                "pending_review_reports": pending_review,
                "open_tasks": open_tasks,
                "observation_snapshots": snapshots,
                "risk_counts": risk_counts,
                "country_counts": country_counts,
                "crop_counts": crop_counts,
            },
            "authoritative": authoritative,
        })

    def _v2_spatial_summary(self) -> None:
        path = Path(__file__).resolve().parents[1] / "results/yieldsat_risk_v1/experiment_07_spatial_maps/field_metrics.csv"
        rows = read_csv(path)
        self._json({
            "fields": [
                {
                    "field_id": row["field_id"],
                    "country": row["country"],
                    "crop": row["crop"],
                    "pixels": int(row["pixels"]),
                    "rmse_spatial": float(row["rmse_spatial"]),
                    "rmse_uniform": float(row["rmse_uniform"]),
                    "rmse_ndvi": float(row["rmse_ndvi"]),
                    "r2_spatial": float(row["r2_spatial"]),
                    "rho_spatial": float(row["rho_spatial"]),
                    "bottom20_auroc": float(row["bottom20_auroc"]),
                }
                for row in rows
            ],
            "count": len(rows),
            "disclosure": "空间指标来自开发挑战集（316 田块）；仅用于田块内相对定位，不是病因图。",
        })

    def _v2_knowledge(self) -> None:
        knowledge = load_risk_knowledge()
        self._json({
            "version": knowledge["version"],
            "scope": knowledge["scope"],
            "governance": knowledge["governance"],
            "entries": [
                {
                    "knowledge_id": entry["knowledge_id"],
                    "title": entry["title"],
                    "countries": entry.get("countries") or [],
                    "crops": entry.get("crops") or [],
                    "required_context": entry.get("required_context") or [],
                    "low_risk_actions": entry["low_risk_actions"],
                    "conditional_actions": entry["conditional_actions"],
                    "source_ids": entry["source_ids"],
                }
                for entry in knowledge["entries"]
            ],
            "sources": knowledge["sources"],
        })

    def _v2_audit_log(self) -> None:
        connection = self._connection()
        try:
            rows = [dict(row) for row in connection.execute(
                "SELECT * FROM audit_log ORDER BY audit_id DESC LIMIT 200"
            ).fetchall()]
            migrations = [dict(row) for row in connection.execute(
                "SELECT * FROM schema_migrations ORDER BY version"
            ).fetchall()]
        finally:
            connection.close()
        self._json({"audit_log": rows, "schema_migrations": migrations})

    def _v2_retrospective_summary(self, user: dict[str, Any]) -> None:
        if ROLE_LEVEL[user["role"]] < ROLE_LEVEL["agronomist"]:
            self._error(HTTPStatus.FORBIDDEN, "回顾审计需要农艺师或管理员角色")
            return
        root = Path(__file__).resolve().parents[1]
        final_report = read_json(root / "results/yieldsat_risk_v1/final_audit/final_report.json")
        p0a_report = read_json(root / "results/yieldsat_risk_v1/post_audit_p0p1/p0a_report.json")
        p0b_report = read_json(root / "results/yieldsat_risk_v1/post_audit_p0p1/p0b_report.json")
        rows = read_csv(root / "results/yieldsat_risk_v1/final_audit/final_predictions.csv")
        self._json({
            "mode": "retrospective_audit",
            "label": "收获后审计（回顾）",
            "disclosure": "本页数据仅用于收获后评估；不得进入前瞻告警、报告或 LLM。",
            "final_audit": {
                "status": final_report["status"],
                "action": final_report["action_alert"],
                "looks": {look: final_report["looks"][look]["metrics"] for look in ("1", "2", "3")},
            },
            "p0a_original_frozen": p0a_report["original_frozen_policy_intervals"],
            "p0b_clean_null": {
                "watch": p0b_report["audit_uniform_policy_comparison"]["dual_tree"]["watch"],
                "action": p0b_report["audit_uniform_policy_comparison"]["dual_tree"]["action"],
            },
            "fields": [
                {
                    "field_id": row["field_id"],
                    "look": int(row["look"]),
                    "prediction": float(row["prediction"]),
                    "target": float(row["target"]),
                    "true_low": int(row["true_low"]),
                    "pvalue": float(row["pvalue"]),
                    "watch": int(row["watch"]),
                    "action_alert": int(row["action_alert"]),
                }
                for row in rows
            ],
        })

    def _v2_export_markdown(self, field_id: str) -> None:
        path = RISK_REPORT_DIR / f"{field_id}.json"
        if not path.exists():
            raise FileNotFoundError(field_id)
        report = read_json(path)
        p = report["professional_analysis"]
        lines = [
            f"# PhysCrop-Risk 田块报告：{report['event_key']}",
            "",
            f"- 状态：{report['status']}",
            f"- 风险等级：{report['risk']['label']}",
            f"- 决策策略：{report['risk']['policy']}",
            f"- 生成时间：{report['generated_at']}",
            "",
            "## 农户摘要",
            "",
            f"**{report['farmer_summary']['headline']}**",
            "",
            report["farmer_summary"]["what_we_observed"],
            "",
            "### 现在可以做什么",
            *[f"- {item}" for item in report["farmer_summary"]["what_to_do_now"]],
            "",
            f"> {report['farmer_summary']['important_note']}",
            "",
            "## 专业分析",
            "",
            p["conclusion"],
            "",
            p["narrative"],
            "",
            "### 三 look 证据轨迹",
            "",
            p["trajectory_assessment"]["narrative"],
            "",
            "### 空间证据",
            "",
            p["spatial_assessment"]["narrative"],
            "",
            "## 分阶段处置",
            "",
            report["action_plan"]["rationale"],
            "",
            "### 24 小时内",
            *[f"- {item}" for item in report["action_plan"]["within_24_hours"]],
            "",
            "### 不要做",
            *[f"- {item}" for item in report["action_plan"]["do_not_do"]],
            "",
            "## 局限性与来源",
            *[f"- {item}" for item in report["limitations"]],
            "",
            *[f"- [{source['title']}]({source['url']})" for source in report["knowledge_citations"]],
            "",
            f"证据哈希：`{report['provenance']['evidence_hash']}`",
        ]
        payload = "\n".join(lines).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/markdown; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="{field_id}_physcrop_risk_report.md"')
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _v2_export_pdf(self, field_id: str) -> None:
        connection = self._connection()
        try:
            row = connection.execute(
                "SELECT report_json FROM reports WHERE event_key=? ORDER BY created_at DESC LIMIT 1",
                (field_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise FileNotFoundError(field_id)
        report = json.loads(row["report_json"])
        payload = build_report_pdf(report)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/pdf")
        encoded_name = quote(f"{field_id}_低产风险报告.pdf")
        self.send_header(
            "Content-Disposition",
            f'attachment; filename="report.pdf"; filename*=UTF-8\'\'{encoded_name}',
        )
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _v2_reports(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        limit = min(int(query.get("limit", ["200"])[0]), 500)
        cursor = int(query.get("cursor", ["0"])[0])
        generator = query.get("generator", [""])[0]
        status = query.get("status", [""])[0]
        country = query.get("country", [""])[0]
        crop = query.get("crop", [""])[0]
        q = query.get("q", [""])[0].strip().lower()
        where = ["e.source_type=?", "e.event_key=r.event_key"]
        params: list[Any] = [RISK_SOURCE_TYPE]
        if generator:
            where.append("r.generator=?")
            params.append(generator)
        if status:
            where.append("r.status=?")
            params.append(status)
        if country:
            where.append("e.country=?")
            params.append(country)
        if crop:
            where.append("e.crop=?")
            params.append(crop)
        if q:
            where.append("r.event_key LIKE ?")
            params.append(f"%{q}%")
        where_sql = " AND ".join(where)
        connection = self._connection()
        try:
            total = connection.execute(
                f"SELECT COUNT(*) FROM reports r JOIN events e ON {where_sql}", params
            ).fetchone()[0]
            rows = connection.execute(
                f"""SELECT r.report_id,r.event_key,r.generator,r.provider,r.model,r.status,r.created_at,
                           e.country,e.crop,e.season_year
                    FROM reports r JOIN events e ON {where_sql}
                    ORDER BY r.created_at DESC LIMIT ? OFFSET ?""",
                [*params, limit, cursor],
            ).fetchall()
        finally:
            connection.close()
        self._json({
            "reports": [
                {
                    "report_id": row["report_id"],
                    "event_key": row["event_key"],
                    "country": row["country"],
                    "crop": row["crop"],
                    "season": str(row["season_year"]),
                    "generator": row["generator"],
                    "provider": row["provider"],
                    "model": row["model"],
                    "status": row["status"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ],
            "count": total,
            "cursor": cursor + len(rows),
            "has_more": cursor + len(rows) < total,
        })

    def _v2_generation_logs(self) -> None:
        connection = self._connection()
        try:
            rows = [dict(row) for row in connection.execute(
                "SELECT * FROM generation_logs ORDER BY log_id DESC LIMIT 100"
            ).fetchall()]
        finally:
            connection.close()
        for row in rows:
            row["stages"] = json.loads(row.pop("stages_json"))
            row["validation"] = json.loads(row.pop("validation_json"))
        self._json({"logs": rows, "count": len(rows)})

    def _v2_create_observation(self, user: dict[str, Any], field_id: str) -> None:
        if not self._v2_field_allowed(user, field_id):
            self._error(HTTPStatus.FORBIDDEN, "无权访问该田块")
            return
        self._v2_risk_evidence(field_id)
        body = self._body()
        snapshot = {
            "schema_version": "physcrop_observation_snapshot_v1",
            "snapshot_id": ulid(),
            "event_key": field_id,
            "observer": user["username"],
            "soil_moisture": str(body.get("soil_moisture", "未记录"))[:100],
            "standing_water": str(body.get("standing_water", "未记录"))[:100],
            "canopy_symptoms": str(body.get("canopy_symptoms", ""))[:500],
            "notes": str(body.get("notes", ""))[:2000],
            "photo_refs": [str(item)[:300] for item in (body.get("photo_refs") or [])][:10],
            "open_questions": [str(item)[:300] for item in (body.get("open_questions") or [])][:10],
            "created_at": utc_now(),
            "immutable": True,
            "does_not_change_model_evidence": True,
        }
        payload = json.dumps(snapshot, ensure_ascii=False, indent=2)
        snapshot_dir = RISK_OBSERVATION_DIR / field_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = snapshot_dir / f"{snapshot['snapshot_id']}.json"
        path.write_text(payload, encoding="utf-8")
        connection = self._connection()
        try:
            connection.execute(
                """INSERT INTO observation_snapshots
                   (snapshot_id,event_key,observer,payload_sha256,payload_uri,status,created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (snapshot["snapshot_id"], field_id, user["username"],
                 hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                 f"risk_observations/{field_id}/{snapshot['snapshot_id']}.json",
                 "pending_review", snapshot["created_at"]),
            )
            connection.execute(
                """INSERT INTO event_timeline(event_key,actor,event_type,title,detail,created_at)
                   VALUES(?,?,?,?,?,?)""",
                (field_id, user["username"], "observation_snapshot", "已保存不可变现场快照",
                 f"土壤水分：{snapshot['soil_moisture']}；积水：{snapshot['standing_water']}",
                 snapshot["created_at"]),
            )
            connection.commit()
            audit(connection, user["username"], "create_observation_snapshot", field_id,
                  {"snapshot_id": snapshot["snapshot_id"]})
        finally:
            connection.close()
        self._json({"snapshot": snapshot}, HTTPStatus.CREATED)

    def _v2_generate_report_job(self, user: dict[str, Any]) -> None:
        body = self._body()
        field_id = str(body.get("event_key", ""))
        provider = str(body.get("provider", "local"))
        idempotency_key = str(body.get("idempotency_key", "")).strip()
        if not self._v2_field_allowed(user, field_id):
            self._error(HTTPStatus.FORBIDDEN, "无权访问该田块")
            return
        if provider not in {"deepseek", "openai", "anthropic", "local"}:
            self._error(HTTPStatus.BAD_REQUEST, "provider 必须是 deepseek/openai/anthropic/local")
            return
        if idempotency_key:
            connection = self._connection()
            try:
                row = connection.execute(
                    "SELECT response_json FROM idempotency_keys WHERE idempotency_key=? AND endpoint=?",
                    (idempotency_key, "/api/v2/reports/generate"),
                ).fetchone()
                if row:
                    self._json({"cached": True, **json.loads(row["response_json"])})
                    return
            finally:
                connection.close()
        job_id = ulid()
        _JOBS[job_id] = {"status": "queued", "job_id": job_id, "event_key": field_id,
                         "provider": provider, "created_at": utc_now(),
                         "stage": "queued", "events": [
                             {"stage": "queued", "message": "任务已排队", "at": utc_now()},
                         ]}

        def run() -> None:
            try:
                _add_job_event(job_id, "preparing", "正在准备证据与知识库…")
                evidence = self._v2_risk_evidence(field_id)
                knowledge = load_risk_knowledge()
                draft = deterministic_risk_report(evidence, knowledge)
                warning = None
                generator = "deterministic_fallback"
                model = "rules-v3"
                if provider != "local":
                    try:
                        _add_job_event(job_id, "calling_llm", f"正在调用大模型（{PROVIDER_ZH_SERVER.get(provider, provider)}）…")
                        candidate, model = call_llm(provider, build_risk_llm_messages(evidence, knowledge, draft))
                        _add_job_event(job_id, "merging", "已收到模型输出，正在合并叙述…")
                        draft = merge_llm_narrative(draft, candidate)
                        generator = "llm_guarded"
                    except Exception as exc:  # noqa: BLE001
                        warning = f"LLM 不可用，已安全降级为规则报告：{exc}"
                        _add_job_event(job_id, "fallback", "大模型调用失败，自动回退规则版本")
                _add_job_event(job_id, "validating", "正在执行事实与安全校验…")
                validation = validate_risk_report(draft, evidence, knowledge)
                if generator == "llm_guarded" and not validation["passes"]:
                    draft = deterministic_risk_report(evidence, knowledge)
                    validation = validate_risk_report(draft, evidence, knowledge)
                    warning = "LLM 报告未通过事实校验，已安全降级为规则报告。"
                    generator = "deterministic_fallback"
                    model = "rules-v3"
                    _add_job_event(job_id, "fallback", "智能润色未通过校验，自动回退规则版本")
                connection = init_db()
                try:
                    path = persist_risk_report(connection, draft, validation, generator, provider, model)
                    audit(connection, user["username"], "generate_risk_report", field_id,
                          {"provider": provider, "model": model, "warning": warning})
                    stages = _JOBS[job_id].get("events", [])
                    connection.execute(
                        """INSERT INTO generation_logs
                           (event_key,report_id,provider,model,generator,stages_json,validation_json,warning,created_at)
                           VALUES(?,?,?,?,?,?,?,?,?)""",
                        (field_id, draft["report_id"], provider, model, generator,
                         json.dumps(stages, ensure_ascii=False),
                         json.dumps(validation, ensure_ascii=False), warning or "", utc_now()),
                    )
                    connection.commit()
                finally:
                    connection.close()
                GENERATION_LOG_DIR.mkdir(parents=True, exist_ok=True)
                write_json(GENERATION_LOG_DIR / f"{draft['report_id']}.json", {
                    "event_key": field_id,
                    "report_id": draft["report_id"],
                    "provider": provider,
                    "model": model,
                    "generator": generator,
                    "stages": stages,
                    "validation": validation,
                    "warning": warning,
                    "created_at": utc_now(),
                })
                _add_job_event(job_id, "done", "报告生成完成")
                result = {"report": draft, "validation": validation, "warning": warning, "path": str(path)}
                _JOBS[job_id].update({"status": "done", "result": result})
                if idempotency_key:
                    connection = init_db()
                    try:
                        connection.execute(
                            """INSERT OR REPLACE INTO idempotency_keys
                               (idempotency_key,endpoint,response_json,created_at) VALUES(?,?,?,?)""",
                            (idempotency_key, "/api/v2/reports/generate",
                             json.dumps({"job_id": job_id}, ensure_ascii=False), utc_now()),
                        )
                        connection.commit()
                    finally:
                        connection.close()
            except Exception as exc:  # noqa: BLE001
                _add_job_event(job_id, "failed", f"生成失败：{exc}")
                _JOBS[job_id].update({"status": "failed", "error": str(exc)})

        _EXECUTOR.submit(run)
        self._json({"job_id": job_id, "status": "queued", "event_key": field_id}, HTTPStatus.ACCEPTED)

    def _v2_job_status(self, job_id: str) -> None:
        job = _JOBS.get(job_id)
        if job is None:
            self._error(HTTPStatus.NOT_FOUND, "任务不存在")
            return
        self._json(job)

    def _v2_get(self, parsed) -> None:
        path = parsed.path
        if path == "/api/v2/health":
            return self._json({"status": "ok", "system": "PhysCrop-Risk report agent v3",
                               "time": utc_now()})
        user = self._v2_user()
        if user is None:
            return
        if path == "/api/v2/dashboard/summary":
            return self._v2_dashboard_summary()
        if path == "/api/v2/alerts":
            return self._v2_alerts(user)
        if path == "/api/v2/reports":
            return self._v2_reports()
        if path == "/api/v2/generation-logs":
            return self._v2_generation_logs()
        if path == "/api/v2/fields/spatial-summary":
            return self._v2_spatial_summary()
        if path == "/api/v2/knowledge":
            return self._v2_knowledge()
        if path == "/api/v2/audit":
            return self._v2_audit_log()
        if path == "/api/v2/retrospective/summary":
            return self._v2_retrospective_summary(user)
        if path.startswith("/api/v2/jobs/"):
            return self._v2_job_status(path[len("/api/v2/jobs/"):])
        if path.startswith("/api/v2/export/report/") and path.endswith(".pdf"):
            return self._v2_export_pdf(path[len("/api/v2/export/report/"):-len(".pdf")])
        if path.startswith("/api/v2/export/report/") and path.endswith(".md"):
            return self._v2_export_markdown(path[len("/api/v2/export/report/"):-len(".md")])
        prefix = "/api/v2/alerts/"
        if path.startswith(prefix):
            return self._v2_alert_detail(user, path[len(prefix):].strip("/"))
        if path == "/api/v2/tasks":
            return self._list_tasks()
        self._error(HTTPStatus.NOT_FOUND, "API v2 endpoint not found")

    def _v2_post(self, parsed) -> None:
        path = parsed.path
        if path == "/api/v2/reports/generate":
            user = self._v2_user("agronomist")
            if user is None:
                return
            return self._v2_generate_report_job(user)
        if path.startswith("/api/v2/reports/") and path.endswith("/review"):
            user = self._v2_user("agronomist")
            if user is None:
                return
            report_id = path[len("/api/v2/reports/"):-len("/review")].strip("/")
            return self._review_report(user, report_id)
        if path.startswith("/api/v2/alerts/") and path.endswith("/observations"):
            user = self._v2_user("farmer")
            if user is None:
                return
            field_id = path[len("/api/v2/alerts/"):-len("/observations")].strip("/")
            return self._v2_create_observation(user, field_id)
        if path.startswith("/api/v2/alerts/") and path.endswith("/assign"):
            user = self._v2_user("agronomist")
            if user is None:
                return
            field_id = path[len("/api/v2/alerts/"):-len("/assign")].strip("/")
            return self._assign_event(user, field_id)
        if path.startswith("/api/v2/alerts/") and path.endswith("/transition"):
            user = self._v2_user("agronomist")
            if user is None:
                return
            field_id = path[len("/api/v2/alerts/"):-len("/transition")].strip("/")
            return self._transition_event(user, field_id)
        self._error(HTTPStatus.NOT_FOUND, "API v2 endpoint not found")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/v2/"):
            try:
                return self._v2_get(parsed)
            except FileNotFoundError:
                self._error(HTTPStatus.NOT_FOUND, "田块或报告不存在")
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return
        if not parsed.path.startswith("/api/"):
            if parsed.path.startswith("/legacy/") or parsed.path == "/legacy":
                legacy_path = parsed.path[len("/legacy"):] or "/"
                self.directory = str(LEGACY_APP_DIR)
                self.path = legacy_path
                return super().do_GET()
            if parsed.path == "/":
                self.path = "/index.html"
            return super().do_GET()
        try:
            if parsed.path == "/api/v1/health":
                return self._json({
                    "status": "ok",
                    "system": "PhysCrop-F-v5 event response agent",
                    "mode": "demo" if self.demo else "authenticated",
                    "time": utc_now(),
                })
            user = self._user()
            if user is None:
                return
            if parsed.path == "/api/v1/session":
                return self._json({"user": user, "demo": self.demo})
            if parsed.path == "/api/v1/events":
                return self._list_events()
            if parsed.path == "/api/v1/commercial/overview":
                return self._commercial_overview()
            if parsed.path == "/api/v1/commercial/map":
                return self._commercial_map()
            if parsed.path == "/api/v1/tasks":
                return self._list_tasks()
            if parsed.path == "/api/v1/knowledge":
                kb = load_knowledge()
                return self._json({"version": kb["version"], "scope": kb["scope"], "sources": kb["sources"]})
            prefix = "/api/v1/events/"
            if parsed.path.startswith(prefix):
                event_key = unquote(parsed.path[len(prefix):])
                return self._event_detail(event_key)
            self._error(HTTPStatus.NOT_FOUND, "API endpoint not found")
        except FileNotFoundError:
            self._error(HTTPStatus.NOT_FOUND, "事件不存在")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _list_events(self) -> None:
        connection = self._connection()
        try:
            rows = connection.execute(
                """SELECT e.*, r.status AS report_status
                   FROM events e LEFT JOIN reports r ON r.report_id=(
                       SELECT r2.report_id FROM reports r2 WHERE r2.event_key=e.event_key
                       ORDER BY r2.updated_at DESC LIMIT 1
                   )
                   WHERE e.source_type=?
                   ORDER BY CASE e.risk_level WHEN 'priority' THEN 1 WHEN 'review' THEN 2
                            WHEN 'watch' THEN 3 ELSE 4 END, e.event_key"""
                , (LEGACY_SOURCE_TYPE,)
            ).fetchall()
            events = []
            seen = set()
            for row in rows:
                if row["event_key"] in seen:
                    continue
                seen.add(row["event_key"])
                evidence = read_json(Path(row["evidence_path"]))
                events.append({
                    "event_key": row["event_key"],
                    "event_id": evidence["event_identity"]["event_id"],
                    "year": evidence["event_identity"]["year"],
                    "patch_id": evidence["event_identity"]["patch_id"],
                    "risk_level": row["risk_level"],
                    "risk_label": evidence["model_assessment"]["risk_label_zh"],
                    "sustained_alert": bool(row["detected"]),
                    "peak_risk": evidence["model_assessment"]["peak_normalized_risk"],
                    "peak_date": evidence["model_assessment"]["peak_date"],
                    "report_status": row["report_status"] or "not_generated",
                    "workflow_status": row["workflow_status"],
                    "workflow_label": WORKFLOW_LABELS.get(row["workflow_status"], row["workflow_status"]),
                    "assignee": row["assignee"],
                    "due_at": row["due_at"],
                })
            self._json({"events": events, "count": len(events)})
        finally:
            connection.close()

    def _event_detail(self, event_key: str) -> None:
        connection = self._connection()
        try:
            event_row = connection.execute(
                "SELECT * FROM events WHERE event_key=?", (event_key,)
            ).fetchone()
            if event_row is None or event_row["source_type"] != LEGACY_SOURCE_TYPE:
                raise FileNotFoundError(event_key)
        finally:
            connection.close()
        evidence = load_event(event_key)
        try:
            report = load_report(event_key)
        except FileNotFoundError:
            report = None
        connection = self._connection()
        try:
            observations = [dict(row) for row in connection.execute(
                "SELECT * FROM field_observations WHERE event_key=? ORDER BY created_at DESC", (event_key,)
            ).fetchall()]
            reviews = []
            if report:
                reviews = [dict(row) for row in connection.execute(
                    "SELECT * FROM reviews WHERE report_id=? ORDER BY created_at DESC", (report["report_id"],)
                ).fetchall()]
            event_row = connection.execute("SELECT * FROM events WHERE event_key=?", (event_key,)).fetchone()
            tasks = [dict(row) for row in connection.execute(
                "SELECT * FROM tasks WHERE event_key=? ORDER BY updated_at DESC", (event_key,)
            ).fetchall()]
            timeline = [dict(row) for row in connection.execute(
                "SELECT * FROM event_timeline WHERE event_key=? ORDER BY created_at DESC", (event_key,)
            ).fetchall()]
        finally:
            connection.close()
        self._json({
            "evidence": evidence, "report": report, "observations": observations, "reviews": reviews,
            "workflow": {
                "status": event_row["workflow_status"],
                "label": WORKFLOW_LABELS.get(event_row["workflow_status"], event_row["workflow_status"]),
                "assignee": event_row["assignee"], "due_at": event_row["due_at"],
                "closed_reason": event_row["closed_reason"],
            },
            "tasks": tasks, "timeline": timeline,
        })

    def _commercial_overview(self) -> None:
        config = read_json(DEMO_CONFIG_PATH)
        connection = self._connection()
        try:
            risk_counts = {row["risk_level"]: row["count"] for row in connection.execute(
                "SELECT risk_level,COUNT(*) AS count FROM events WHERE source_type=? GROUP BY risk_level",
                (LEGACY_SOURCE_TYPE,),
            ).fetchall()}
            workflow_counts = {status: 0 for status in WORKFLOW_LABELS}
            workflow_counts.update({row["workflow_status"]: row["count"] for row in connection.execute(
                "SELECT workflow_status,COUNT(*) AS count FROM events WHERE source_type=? GROUP BY workflow_status",
                (LEGACY_SOURCE_TYPE,),
            ).fetchall()})
            event_count = connection.execute(
                "SELECT COUNT(*) FROM events WHERE source_type=?", (LEGACY_SOURCE_TYPE,)
            ).fetchone()[0]
            detected = connection.execute(
                "SELECT COUNT(*) FROM events WHERE source_type=? AND detected=1", (LEGACY_SOURCE_TYPE,)
            ).fetchone()[0]
            open_tasks = connection.execute("SELECT COUNT(*) FROM tasks WHERE status!='completed'").fetchone()[0]
            approved = connection.execute(
                "SELECT COUNT(DISTINCT event_key) FROM reports WHERE status='approved'"
            ).fetchone()[0]
        finally:
            connection.close()

        table = {row["method"]: row for row in read_csv(MAIN_TABLE_PATH)}
        phys = table["PhysCrop-F-v5"]
        learning = [row for name, row in table.items() if name not in {"PhysCrop-F-v5", "NDVI Threshold"}]
        best_tpr = max(float(row["tpr_at_5fpr_mean"]) for row in learning)
        best_aupr = max(float(row["aupr_mean"]) for row in learning)
        best_filling = min(float(row["filling_fpr_at_5fpr_mean"]) for row in learning)
        metrics = {
            "tpr_at_5fpr": round(float(phys["tpr_at_5fpr_mean"]), 2),
            "aupr": round(float(phys["aupr_mean"]), 2),
            "filling_fpr_at_5fpr": round(float(phys["filling_fpr_at_5fpr_mean"]), 2),
            "conditional_lead_days": round(float(phys["lead_at_5fpr_mean"]), 2),
            "event_recall_at_5fpr": round(float(phys["event_recall_at_5fpr_mean"]), 2),
            "tpr_gain_vs_best_learning_baseline_pp": round(float(phys["tpr_at_5fpr_mean"]) - best_tpr, 2),
            "aupr_gain_vs_best_learning_baseline_pp": round(float(phys["aupr_mean"]) - best_aupr, 2),
            "filling_fpr_reduction_vs_best_learning_baseline_pp": round(best_filling - float(phys["filling_fpr_at_5fpr_mean"]), 2),
        }
        self._json({
            "config": config,
            "portfolio": {
                "event_count": event_count, "consensus_detected_events": detected,
                "risk_counts": risk_counts, "workflow_counts": workflow_counts,
                "open_tasks": open_tasks, "approved_reports": approved,
            },
            "model_metrics": metrics,
            "metric_disclosure": "指标来自protocol_v2_clean controlled benchmark；优势比较仅针对表中的学习型基线。",
        })

    def _commercial_map(self) -> None:
        points = []
        for path in sorted(EVIDENCE_DIR.glob("*.json")):
            evidence = read_json(path)
            identity = evidence["event_identity"]
            assessment = evidence["model_assessment"]
            points.append({
                "event_key": evidence["event_key"], "event_id": identity["event_id"],
                "row": identity.get("row"), "col": identity.get("col"),
                "year": identity["year"], "patch_id": identity["patch_id"],
                "field_name": identity.get("field_name", f"Patch {identity['patch_id']}"),
                "risk_level": assessment["risk_level"], "risk_label": assessment["risk_label_zh"],
                "peak_risk": assessment["peak_normalized_risk"], "sustained_alert": assessment["sustained_alert"],
            })
        valid = [point for point in points if point["row"] is not None and point["col"] is not None]
        bounds = {
            "min_row": min(point["row"] for point in valid), "max_row": max(point["row"] for point in valid),
            "min_col": min(point["col"] for point in valid), "max_col": max(point["col"] for point in valid),
        } if valid else None
        self._json({
            "points": points, "bounds": bounds, "map_type": "controlled_spatial_index",
            "disclosure": read_json(DEMO_CONFIG_PATH)["spatial_disclosure"],
            "uav_asset": "/assets/uav_anda_tile_overview.png",
        })

    def _list_tasks(self) -> None:
        connection = self._connection()
        try:
            tasks = [dict(row) for row in connection.execute(
                "SELECT * FROM tasks ORDER BY CASE status WHEN 'open' THEN 1 WHEN 'in_progress' THEN 2 ELSE 3 END, due_at"
            ).fetchall()]
        finally:
            connection.close()
        self._json({"tasks": tasks, "count": len(tasks)})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/v2/"):
                return self._v2_post(parsed)
            if parsed.path == "/api/v1/reports/generate":
                user = self._user("agronomist")
                if user is None:
                    return
                return self._generate_report(user)
            if parsed.path.startswith("/api/v1/reports/") and parsed.path.endswith("/review"):
                user = self._user("agronomist")
                if user is None:
                    return
                report_id = unquote(parsed.path[len("/api/v1/reports/"):-len("/review")].strip("/"))
                return self._review_report(user, report_id)
            if parsed.path.startswith("/api/v1/events/") and parsed.path.endswith("/observations"):
                user = self._user("farmer")
                if user is None:
                    return
                event_key = unquote(parsed.path[len("/api/v1/events/"):-len("/observations")].strip("/"))
                return self._add_observation(user, event_key)
            if parsed.path.startswith("/api/v1/events/") and parsed.path.endswith("/assign"):
                user = self._user("agronomist")
                if user is None:
                    return
                event_key = unquote(parsed.path[len("/api/v1/events/"):-len("/assign")].strip("/"))
                return self._assign_event(user, event_key)
            if parsed.path.startswith("/api/v1/events/") and parsed.path.endswith("/transition"):
                user = self._user("agronomist")
                if user is None:
                    return
                event_key = unquote(parsed.path[len("/api/v1/events/"):-len("/transition")].strip("/"))
                return self._transition_event(user, event_key)
            if parsed.path.startswith("/api/v1/tasks/") and parsed.path.endswith("/complete"):
                user = self._user("farmer")
                if user is None:
                    return
                task_id = int(unquote(parsed.path[len("/api/v1/tasks/"):-len("/complete")].strip("/")))
                return self._complete_task(user, task_id)
            self._error(HTTPStatus.NOT_FOUND, "API endpoint not found")
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except FileNotFoundError:
            self._error(HTTPStatus.NOT_FOUND, "事件或报告不存在")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _assign_event(self, user: dict[str, str], event_key: str) -> None:
        evidence = load_event(event_key)
        body = self._body()
        assignee = str(body.get("assignee", "")).strip()[:100]
        due_at = str(body.get("due_at", "")).strip()[:40]
        notes = str(body.get("notes", "")).strip()[:1000]
        if not assignee or not due_at:
            raise ValueError("assignee and due_at are required")
        connection = self._connection()
        try:
            row = connection.execute("SELECT * FROM events WHERE event_key=?", (event_key,)).fetchone()
            if row is None:
                raise FileNotFoundError(event_key)
            if row["workflow_status"] not in {"new", "assigned", "in_field"}:
                raise ValueError("当前状态不允许分派或重新分派现场任务")
            now = utc_now()
            title = f"现场核验 {evidence['event_identity']['field_name']}"
            existing_task = connection.execute(
                "SELECT task_id FROM tasks WHERE event_key=? AND status!='completed' ORDER BY task_id DESC LIMIT 1",
                (event_key,),
            ).fetchone()
            if existing_task:
                task_id = existing_task["task_id"]
                connection.execute(
                    "UPDATE tasks SET assignee=?,due_at=?,status='open',notes=?,updated_at=? WHERE task_id=?",
                    (assignee, due_at, notes, now, task_id),
                )
            else:
                task_cursor = connection.execute(
                    """INSERT INTO tasks(event_key,title,assignee,due_at,status,priority,notes,created_by,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (event_key, title, assignee, due_at, "open", evidence["model_assessment"]["risk_level"],
                     notes, user["username"], now, now),
                )
                task_id = task_cursor.lastrowid
            connection.execute(
                "UPDATE events SET workflow_status='assigned',assignee=?,due_at=?,updated_at=? WHERE event_key=?",
                (assignee, due_at, now, event_key),
            )
            connection.execute(
                """INSERT INTO event_timeline(event_key,actor,event_type,title,detail,created_at)
                   VALUES(?,?,?,?,?,?)""",
                (event_key, user["username"], "assignment", "已分派现场核验任务",
                 f"负责人：{assignee}；截止：{due_at}。{notes}", now),
            )
            connection.commit()
            audit(connection, user["username"], "assign_event", event_key, {"task_id": task_id, "assignee": assignee})
        finally:
            connection.close()
        self._json({"assigned": True, "event_key": event_key, "task_id": task_id}, HTTPStatus.CREATED)

    def _transition_event(self, user: dict[str, str], event_key: str) -> None:
        load_event(event_key)
        body = self._body()
        target = str(body.get("status", ""))
        comment = str(body.get("comment", "")).strip()[:1500]
        if target not in WORKFLOW_LABELS:
            raise ValueError("Unknown workflow status")
        connection = self._connection()
        try:
            row = connection.execute("SELECT * FROM events WHERE event_key=?", (event_key,)).fetchone()
            if row is None:
                raise FileNotFoundError(event_key)
            current = row["workflow_status"]
            if target not in ALLOWED_TRANSITIONS.get(current, set()):
                raise ValueError(f"Invalid transition: {current} -> {target}")
            observation_count = connection.execute(
                "SELECT COUNT(*) FROM field_observations WHERE event_key=?", (event_key,)
            ).fetchone()[0]
            active_task_count = connection.execute(
                "SELECT COUNT(*) FROM tasks WHERE event_key=? AND status!='completed'", (event_key,)
            ).fetchone()[0]
            latest_report = connection.execute(
                "SELECT status FROM reports WHERE event_key=? ORDER BY updated_at DESC LIMIT 1", (event_key,)
            ).fetchone()
            if target in {"assigned", "in_field"} and active_task_count == 0:
                raise ValueError("进入现场核验流程前必须创建有效任务")
            if target == "feedback_received" and observation_count == 0:
                raise ValueError("进入已反馈状态前必须提交现场记录")
            if target == "under_review" and (latest_report is None or latest_report["status"] != "approved"):
                raise ValueError("进入农艺审核状态前报告必须完成人工批准")
            if target == "closed" and not comment:
                raise ValueError("关闭事件必须填写关闭依据")
            if target == "closed" and observation_count == 0:
                raise ValueError("关闭事件前必须有现场反馈")
            if target == "closed" and (latest_report is None or latest_report["status"] != "approved"):
                raise ValueError("关闭事件前报告必须完成人工批准")
            now = utc_now()
            closed_reason = comment if target == "closed" else row["closed_reason"]
            connection.execute(
                "UPDATE events SET workflow_status=?,closed_reason=?,updated_at=? WHERE event_key=?",
                (target, closed_reason, now, event_key),
            )
            if target == "in_field":
                connection.execute(
                    "UPDATE tasks SET status='in_progress',updated_at=? WHERE event_key=? AND status='open'",
                    (now, event_key),
                )
            if target in {"feedback_received", "closed"}:
                connection.execute(
                    "UPDATE tasks SET status='completed',updated_at=? WHERE event_key=? AND status!='completed'",
                    (now, event_key),
                )
            connection.execute(
                """INSERT INTO event_timeline(event_key,actor,event_type,title,detail,created_at)
                   VALUES(?,?,?,?,?,?)""",
                (event_key, user["username"], "workflow", f"状态更新为{WORKFLOW_LABELS[target]}", comment, now),
            )
            connection.commit()
            audit(connection, user["username"], "transition_event", event_key, {"from": current, "to": target})
        finally:
            connection.close()
        self._json({"event_key": event_key, "status": target, "label": WORKFLOW_LABELS[target]})

    def _complete_task(self, user: dict[str, str], task_id: int) -> None:
        body = self._body()
        note = str(body.get("note", "")).strip()[:1000]
        connection = self._connection()
        try:
            task = connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if task is None:
                raise FileNotFoundError(str(task_id))
            now = utc_now()
            connection.execute("UPDATE tasks SET status='completed',notes=?,updated_at=? WHERE task_id=?", (note, now, task_id))
            connection.execute(
                """INSERT INTO event_timeline(event_key,actor,event_type,title,detail,created_at)
                   VALUES(?,?,?,?,?,?)""",
                (task["event_key"], user["username"], "task", "现场任务已完成", note, now),
            )
            connection.commit()
            audit(connection, user["username"], "complete_task", str(task_id), {"event_key": task["event_key"]})
        finally:
            connection.close()
        self._json({"task_id": task_id, "status": "completed"})

    def _generate_report(self, user: dict[str, str]) -> None:
        body = self._body()
        event_key = str(body.get("event_key", ""))
        provider = str(body.get("provider", "deepseek"))
        if provider not in {"deepseek", "openai", "anthropic", "local"}:
            raise ValueError("provider must be deepseek, openai, anthropic, or local")
        evidence = load_event(event_key)
        knowledge = load_knowledge()
        draft = deterministic_report(evidence, knowledge)
        warning = None
        model = "rules-v1"
        generator = "deterministic_fallback"
        if provider != "local":
            try:
                candidate, model = call_llm(provider, build_llm_messages(evidence, knowledge, draft))
                draft = merge_llm_narrative(draft, candidate)
                generator = "llm_guarded"
            except Exception as exc:  # noqa: BLE001
                warning = f"LLM 不可用，已安全降级为规则报告：{exc}"
                provider = "local"
        validation = validate_report(draft, evidence, knowledge)
        if generator == "llm_guarded" and not validation["passes"]:
            failed_errors = list(validation["errors"])
            draft = deterministic_report(evidence, knowledge)
            validation = validate_report(draft, evidence, knowledge)
            warning = "LLM 报告未通过事实校验，已安全降级为规则报告：" + "；".join(failed_errors[:4])
            provider = "local"
            model = "rules-v1"
            generator = "deterministic_fallback"
        connection = self._connection()
        try:
            persist_report(connection, draft, validation, generator, provider, model)
            audit(connection, user["username"], "generate_report", event_key, {
                "provider": provider, "model": model, "validation": validation, "warning": warning,
            })
        finally:
            connection.close()
        log_path = LOG_DIR / f"{event_key}_{draft['report_id']}.json"
        write_json(log_path, {
            "event_key": event_key, "report_id": draft["report_id"], "provider": provider,
            "model": model, "validation": validation, "warning": warning, "created_at": utc_now(),
        })
        self._json({"report": draft, "validation": validation, "warning": warning})

    def _review_report(self, user: dict[str, str], report_id: str) -> None:
        body = self._body()
        decision = body.get("decision")
        if decision not in {"approved", "rejected", "needs_revision"}:
            raise ValueError("decision must be approved, rejected, or needs_revision")
        comment = str(body.get("comment", "")).strip()[:2000]
        if decision != "approved" and not comment:
            raise ValueError("驳回或退回修改时必须填写说明")
        connection = self._connection()
        try:
            row = connection.execute("SELECT * FROM reports WHERE report_id=?", (report_id,)).fetchone()
            if row is None:
                raise FileNotFoundError(report_id)
            report = json.loads(row["report_json"])
            report["status"] = decision
            report["review"] = {"decision": decision, "reviewer": user["username"], "comment": comment}
            connection.execute(
                "INSERT INTO reviews(report_id,reviewer,decision,comment,created_at) VALUES(?,?,?,?,?)",
                (report_id, user["username"], decision, comment, utc_now()),
            )
            connection.execute(
                "UPDATE reports SET status=?,report_json=?,updated_at=? WHERE report_id=?",
                (decision, json.dumps(report, ensure_ascii=False), utc_now(), report_id),
            )
            if decision == "approved":
                connection.execute(
                    """UPDATE events SET workflow_status='under_review',updated_at=?
                       WHERE event_key=? AND workflow_status='feedback_received'""",
                    (utc_now(), report["event_key"]),
                )
            connection.execute(
                """INSERT INTO event_timeline(event_key,actor,event_type,title,detail,created_at)
                   VALUES(?,?,?,?,?,?)""",
                (report["event_key"], user["username"], "review", f"报告审核：{decision}", comment, utc_now()),
            )
            connection.commit()
            payload_uri = row["payload_uri"] or f"reports/{report['event_key']}.json"
            write_json(AGENT_ROOT / payload_uri, report)
            audit(connection, user["username"], "review_report", report_id, {"decision": decision})
        finally:
            connection.close()
        self._json({"report": report})

    def _add_observation(self, user: dict[str, str], event_key: str) -> None:
        load_event(event_key)
        body = self._body()
        values = {
            "soil_moisture": str(body.get("soil_moisture", "未记录"))[:100],
            "standing_water": str(body.get("standing_water", "未记录"))[:100],
            "canopy_symptoms": str(body.get("canopy_symptoms", "未记录"))[:500],
            "notes": str(body.get("notes", ""))[:2000],
        }
        connection = self._connection()
        try:
            connection.execute(
                """INSERT INTO field_observations
                   (event_key,observer,soil_moisture,standing_water,canopy_symptoms,notes,created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (event_key, user["username"], values["soil_moisture"], values["standing_water"],
                 values["canopy_symptoms"], values["notes"], utc_now()),
            )
            current = connection.execute(
                "SELECT workflow_status FROM events WHERE event_key=?", (event_key,)
            ).fetchone()[0]
            if current in {"assigned", "in_field"}:
                connection.execute(
                    "UPDATE events SET workflow_status='feedback_received',updated_at=? WHERE event_key=?",
                    (utc_now(), event_key),
                )
                connection.execute(
                    "UPDATE tasks SET status='completed',updated_at=? WHERE event_key=? AND status!='completed'",
                    (utc_now(), event_key),
                )
            connection.execute(
                """INSERT INTO event_timeline(event_key,actor,event_type,title,detail,created_at)
                   VALUES(?,?,?,?,?,?)""",
                (event_key, user["username"], "observation", "已提交现场反馈",
                 f"土壤水分：{values['soil_moisture']}；积水：{values['standing_water']}；冠层：{values['canopy_symptoms']}",
                 utc_now()),
            )
            connection.commit()
            audit(connection, user["username"], "add_field_observation", event_key, values)
        finally:
            connection.close()
        self._json({"saved": True, "event_key": event_key}, HTTPStatus.CREATED)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--demo", action="store_true", help="Localhost-only demo without bearer auth")
    args = parser.parse_args()
    if args.demo and args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("--demo may only bind to localhost")
    users = configured_users()
    if not args.demo and not users:
        raise SystemExit("Set PHYSCROP_AGENT_USERS_JSON or run localhost with --demo")
    app_dir = V2_APP_DIR if V2_APP_DIR.is_dir() else LEGACY_APP_DIR
    handler = partial(AgentHandler, directory=str(app_dir), demo=args.demo, users=users)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"PhysCrop Agent v2 running at http://{args.host}:{args.port}")
    print(f"Serving UI: {app_dir.name} (legacy at /legacy/)")
    print("Mode:", "LOCAL DEMO" if args.demo else "AUTHENTICATED")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
