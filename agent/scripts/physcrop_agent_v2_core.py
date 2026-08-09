"""Core services for the PhysCrop-F-v5 event response agent.

The module deliberately keeps detection, evidence, agronomy retrieval, language
generation, and approval as separate layers. The LLM may rewrite explanations,
but it cannot change model facts, risk gates, or the approved action inventory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION = ROOT / "results" / "protocol_v2_clean" / "controlled_confirmation_v2"
AGENT_ROOT = ROOT / "baseline_report" / "wacv" / "report_agent_v2"
EVIDENCE_DIR = AGENT_ROOT / "event_evidence"
REPORT_DIR = AGENT_ROOT / "reports"
LOG_DIR = AGENT_ROOT / "logs"
KB_PATH = AGENT_ROOT / "knowledge" / "agronomy_knowledge_zh.json"
DB_PATH = AGENT_ROOT / "physcrop_agent_v2.sqlite3"

SCORES_PATH = CONFIRMATION / "physcrop_f_v5_confirmation_scores.csv"
RUNS_PATH = CONFIRMATION / "wacv_tables_v5" / "wacv_v5_main_runs.csv"
MODEL_NAME = "PhysCrop-F-v5"
MODEL_SEEDS = (0, 1, 2)
INJECTION_SEEDS = (200, 201, 202)
REPORT_SCHEMA_VERSION = "physcrop_event_report_v2.1"
EVIDENCE_SCHEMA_VERSION = "physcrop_event_evidence_v2"

FEATURE_GUIDE = {
    "ndvi_area_005": ("NDVI变化面积", "冠层活力变化在地块中的覆盖范围"),
    "ndmi_area_003": ("NDMI变化面积", "水分相关光谱变化在地块中的覆盖范围"),
    "ndvi_lcc_003": ("冠层连通区域", "冠层变化是否形成连续空间斑块"),
    "ndmi_lcc_002": ("水分连通区域", "水分相关变化是否形成连续空间斑块"),
    "joint_lcc_002": ("联合连通区域", "冠层与水分代理共同变化的连通范围"),
    "joint_coherence": ("空间一致性", "多项变化证据在空间上是否相互支持"),
    "ndvi_top_drop": ("NDVI局部降幅", "变化最明显区域的冠层代理降幅"),
    "ndmi_top_drop": ("NDMI局部降幅", "变化最明显区域的水分代理降幅"),
}

RISK_LABELS = {
    "monitor": "常规监测",
    "watch": "关注复查",
    "review": "建议现场核验",
    "priority": "优先现场核验",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=False)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ulid() -> str:
    """Crockford-base32 ULID: 48-bit millisecond timestamp + 80-bit randomness."""
    timestamp_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    random_bits = int.from_bytes(os.urandom(10), "big")
    value = (timestamp_ms << 80) | random_bits
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    chars = []
    for _ in range(26):
        chars.append(alphabet[value & 31])
        value >>= 5
    return "".join(reversed(chars))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def rounded(value: float, digits: int = 4) -> float | None:
    return round(value, digits) if math.isfinite(value) else None


def percentile_rank(values: Iterable[float], target: float) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return math.nan
    return 100.0 * sum(value <= target for value in clean) / len(clean)


MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        ALTER TABLE events ADD COLUMN payload_uri TEXT NOT NULL DEFAULT '';
        ALTER TABLE events ADD COLUMN payload_sha256 TEXT NOT NULL DEFAULT '';
        ALTER TABLE events ADD COLUMN source_type TEXT NOT NULL DEFAULT 'anda_controlled_v2';
        ALTER TABLE events ADD COLUMN schema_version TEXT NOT NULL DEFAULT 'physcrop_event_evidence_v2';
        ALTER TABLE events ADD COLUMN country TEXT NOT NULL DEFAULT '';
        ALTER TABLE events ADD COLUMN crop TEXT NOT NULL DEFAULT '';
        ALTER TABLE events ADD COLUMN season_year INTEGER;
        ALTER TABLE events ADD COLUMN watch_original INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE events ADD COLUMN action_original INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE events ADD COLUMN watch_clean INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE events ADD COLUMN action_clean INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE events ADD COLUMN current_look INTEGER;
        ALTER TABLE reports ADD COLUMN payload_uri TEXT NOT NULL DEFAULT '';
        ALTER TABLE reports ADD COLUMN payload_sha256 TEXT NOT NULL DEFAULT '';
        ALTER TABLE reports ADD COLUMN report_version TEXT NOT NULL DEFAULT '1';
        ALTER TABLE reports ADD COLUMN parent_report_id TEXT NOT NULL DEFAULT '';
        ALTER TABLE reports ADD COLUMN superseded_by TEXT NOT NULL DEFAULT '';
        """,
    ),
    (
        2,
        """
        CREATE TABLE IF NOT EXISTS organizations (
            organization_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS farms (
            farm_id TEXT PRIMARY KEY,
            organization_id TEXT NOT NULL,
            name TEXT NOT NULL,
            country TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS fields (
            field_id TEXT PRIMARY KEY,
            farm_id TEXT NOT NULL,
            organization_id TEXT NOT NULL,
            country TEXT NOT NULL,
            crop TEXT NOT NULL,
            year INTEGER NOT NULL,
            spatial_available INTEGER NOT NULL DEFAULT 0,
            inserted_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS observation_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            event_key TEXT NOT NULL,
            observer TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL,
            payload_uri TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending_review',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS idempotency_keys (
            idempotency_key TEXT PRIMARY KEY,
            endpoint TEXT NOT NULL,
            response_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS notification_queue (
            notification_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_key TEXT NOT NULL,
            notification_type TEXT NOT NULL,
            channel TEXT NOT NULL DEFAULT 'in_app',
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL
        );
        """,
    ),
    (
        3,
        """
        CREATE TABLE IF NOT EXISTS generation_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_key TEXT NOT NULL,
            report_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            generator TEXT NOT NULL,
            stages_json TEXT NOT NULL,
            validation_json TEXT NOT NULL,
            warning TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_genlogs_event ON generation_logs(event_key, created_at);
        """,
    ),
]


def _apply_migrations(connection: sqlite3.Connection) -> None:
    applied = {row[0] for row in connection.execute("SELECT version FROM schema_migrations")}
    for version, sql in MIGRATIONS:
        if version in applied:
            continue
        connection.executescript(sql)
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES(?,?)",
            (version, utc_now()),
        )
    connection.commit()


def init_db(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS events (
            event_key TEXT PRIMARY KEY,
            evidence_path TEXT NOT NULL,
            evidence_hash TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            detected INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reports (
            report_id TEXT PRIMARY KEY,
            event_key TEXT NOT NULL,
            status TEXT NOT NULL,
            generator TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            report_json TEXT NOT NULL,
            evidence_hash TEXT NOT NULL,
            knowledge_version TEXT NOT NULL,
            validation_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_reports_event ON reports(event_key, created_at);
        CREATE TABLE IF NOT EXISTS reviews (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id TEXT NOT NULL,
            reviewer TEXT NOT NULL,
            decision TEXT NOT NULL,
            comment TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS field_observations (
            observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_key TEXT NOT NULL,
            observer TEXT NOT NULL,
            soil_moisture TEXT NOT NULL,
            standing_water TEXT NOT NULL,
            canopy_symptoms TEXT NOT NULL,
            notes TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS audit_log (
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor TEXT NOT NULL,
            action TEXT NOT NULL,
            resource TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tasks (
            task_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_key TEXT NOT NULL,
            title TEXT NOT NULL,
            assignee TEXT NOT NULL,
            due_at TEXT NOT NULL,
            status TEXT NOT NULL,
            priority TEXT NOT NULL,
            notes TEXT NOT NULL,
            created_by TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_event ON tasks(event_key, updated_at);
        CREATE TABLE IF NOT EXISTS event_timeline (
            timeline_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_key TEXT NOT NULL,
            actor TEXT NOT NULL,
            event_type TEXT NOT NULL,
            title TEXT NOT NULL,
            detail TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_timeline_event ON event_timeline(event_key, created_at);
        """
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS schema_migrations (
               version INTEGER PRIMARY KEY,
               applied_at TEXT NOT NULL
           )"""
    )
    _apply_migrations(connection)
    event_columns = {row[1] for row in connection.execute("PRAGMA table_info(events)")}
    event_migrations = {
        "workflow_status": "TEXT NOT NULL DEFAULT 'new'",
        "assignee": "TEXT NOT NULL DEFAULT ''",
        "due_at": "TEXT NOT NULL DEFAULT ''",
        "closed_reason": "TEXT NOT NULL DEFAULT ''",
    }
    for column, definition in event_migrations.items():
        if column not in event_columns:
            connection.execute(f"ALTER TABLE events ADD COLUMN {column} {definition}")
    connection.commit()
    return connection


def audit(connection: sqlite3.Connection, actor: str, action: str, resource: str,
          metadata: dict[str, Any] | None = None) -> None:
    connection.execute(
        "INSERT INTO audit_log(actor,action,resource,metadata_json,created_at) VALUES(?,?,?,?,?)",
        (actor, action, resource, json.dumps(metadata or {}, ensure_ascii=False), utc_now()),
    )
    connection.commit()


def _feature_rows(seed: int) -> dict[str, dict[str, str]]:
    path = CONFIRMATION / f"physcrop_f_v5_spatial_features_seed{seed}.csv"
    return {row["sample_id"]: row for row in read_csv(path)}


def _thresholds() -> dict[tuple[int, int], float]:
    thresholds: dict[tuple[int, int], float] = {}
    for row in read_csv(RUNS_PATH):
        if row["method"] == MODEL_NAME:
            thresholds[(int(row["model_seed"]), int(row["injection_seed"]))] = float(
                row["threshold_5fpr"]
            )
    expected = {(model, injection) for model in MODEL_SEEDS for injection in INJECTION_SEEDS}
    missing = expected - set(thresholds)
    if missing:
        raise ValueError(f"Missing v5 thresholds: {sorted(missing)}")
    return thresholds


def _first_sustained_alert(trajectory: list[dict[str, Any]], onset: str, severe: str) -> str | None:
    eligible = [point for point in trajectory if onset <= point["date"] <= severe]
    for left, right in zip(eligible[:-1], eligible[1:]):
        if left["normalized_risk"] >= 1.0 and right["normalized_risk"] >= 1.0:
            return left["date"]
    return None


def _risk_gate(detected: bool, peak_ratio: float) -> str:
    if detected and peak_ratio >= 1.5:
        return "priority"
    if detected:
        return "review"
    if peak_ratio >= 1.0:
        return "watch"
    return "monitor"


def generate_all_evidence() -> list[Path]:
    """Build one immutable evidence package per controlled test event."""
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = _thresholds()
    scores = read_csv(SCORES_PATH)
    by_injection: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in scores:
        seed = int(row["injection_seed"])
        if seed in INJECTION_SEEDS and row["split"] == "test":
            by_injection[seed].append(row)

    output_paths: list[Path] = []
    connection = init_db()
    try:
        for injection_seed in INJECTION_SEEDS:
            feature_map = _feature_rows(injection_seed)
            feature_names = next(iter(feature_map.values())).keys() - {"sample_id"}
            feature_reference = {
                name: [as_float(row[name]) for row in feature_map.values()]
                for name in feature_names
            }
            event_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in by_injection[injection_seed]:
                if row.get("event_id"):
                    event_rows[row["event_id"]].append(row)

            schedule_path = CONFIRMATION / f"controlled_schedule_seed{injection_seed}.csv"
            schedule_by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in read_csv(schedule_path):
                if row["split"] == "test":
                    schedule_by_event[row["event_id"]].append(row)

            for event_id, rows in sorted(event_rows.items()):
                rows_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
                for row in rows:
                    rows_by_sample[row["sample_id"]].append(row)
                reference = next(iter(rows_by_sample.values()))[0]
                trajectory: list[dict[str, Any]] = []
                for sample_id, sample_rows in rows_by_sample.items():
                    sample_rows.sort(key=lambda item: int(item["model_seed"]))
                    ratios = [
                        float(row["score"]) / thresholds[(int(row["model_seed"]), injection_seed)]
                        for row in sample_rows
                    ]
                    raw_scores = [float(row["score"]) for row in sample_rows]
                    point: dict[str, Any] = {
                        "sample_id": sample_id,
                        "date": sample_rows[0]["date"],
                        "normalized_risk": rounded(statistics.mean(ratios)),
                        "normalized_risk_min": rounded(min(ratios)),
                        "normalized_risk_max": rounded(max(ratios)),
                        "model_votes": sum(value >= 1.0 for value in ratios),
                        "model_count": len(ratios),
                        "mean_raw_score": rounded(statistics.mean(raw_scores)),
                    }
                    features = feature_map.get(sample_id, {})
                    point["spatial_features"] = {
                        name: rounded(as_float(features.get(name))) for name in sorted(feature_names)
                    }
                    trajectory.append(point)
                trajectory.sort(key=lambda item: item["date"])

                schedule = schedule_by_event[event_id]
                onset = schedule[0]["onset_date"]
                severe = schedule[0]["severe_date"]
                alert_date = _first_sustained_alert(trajectory, onset, severe)
                detected = alert_date is not None
                lead_days = None
                if alert_date:
                    lead_days = (datetime.fromisoformat(severe) - datetime.fromisoformat(alert_date)).days
                peak = max(trajectory, key=lambda item: item["normalized_risk"])
                peak_features = peak["spatial_features"]
                percentiles = {
                    name: rounded(percentile_rank(feature_reference[name], as_float(value)), 1)
                    for name, value in peak_features.items()
                    if value is not None
                }
                risk_level = _risk_gate(detected, float(peak["normalized_risk"]))
                event_key = f"{event_id}_seed{injection_seed}"
                source_paths = [
                    SCORES_PATH,
                    RUNS_PATH,
                    CONFIRMATION / f"physcrop_f_v5_spatial_features_seed{injection_seed}.csv",
                    schedule_path,
                ]
                evidence: dict[str, Any] = {
                    "schema_version": EVIDENCE_SCHEMA_VERSION,
                    "event_key": event_key,
                    "generated_at": utc_now(),
                    "mode": "controlled_audit",
                    "event_identity": {
                        "event_id": event_id,
                        "injection_seed": injection_seed,
                        "year": int(reference["year"]),
                        "patch_id": int(reference["patch_id"]),
                        "row": int(schedule[0]["row"]),
                        "col": int(schedule[0]["col"]),
                        "split": "test",
                        "region": "安达，黑龙江",
                        "crop_type": "unknown",
                        "farm_id": "ANDA-CONTROLLED-DEMO",
                        "field_name": f"演示网格 R{schedule[0]['row']}-C{schedule[0]['col']}",
                        "controlled_demo": True,
                    },
                    "model_assessment": {
                        "model": MODEL_NAME,
                        "operating_point": "validation-normal threshold at 5% FPR",
                        "model_seeds": list(MODEL_SEEDS),
                        "aggregation": "mean of per-seed score/threshold ratios",
                        "event_rule": "two consecutive observations at normalized_risk >= 1 between onset and severe",
                        "sustained_alert": detected,
                        "first_sustained_alert_date": alert_date,
                        "lead_to_severe_days": lead_days,
                        "peak_date": peak["date"],
                        "peak_normalized_risk": peak["normalized_risk"],
                        "peak_model_votes": peak["model_votes"],
                        "risk_level": risk_level,
                        "risk_label_zh": RISK_LABELS[risk_level],
                    },
                    "evidence_summary": {
                        "peak_sample_id": peak["sample_id"],
                        "peak_spatial_features": peak_features,
                        "peak_feature_percentiles_within_seed": percentiles,
                        "interpretation_boundary": "这些特征描述变化的强度和空间一致性，不识别具体病因。",
                    },
                    "trajectory": trajectory,
                    "data_quality": {
                        "status": "limited_metadata" if reference.get("event_id") else "review",
                        "observation_count": len(trajectory),
                        "model_seed_count": len(MODEL_SEEDS),
                        "feature_coverage_percent": rounded(
                            100.0 * sum(bool(point["spatial_features"]) for point in trajectory) / len(trajectory), 1
                        ),
                        "weather_trace_available": False,
                        "field_observation_available": False,
                        "crop_type_available": False,
                        "limitations": [
                            "当前事件来自受控注入测试，不能代表真实病因标签。",
                            "证据包未包含逐日 ERA5 原始天气值，不能把风险归因于某次天气过程。",
                            "缺少作物类型、生育期、土壤水分和田间症状，农业建议仅用于低风险排查。"
                        ],
                    },
                    "audit_context": {
                        "controlled_only": True,
                        "injected_stress_type": schedule[0]["stress_type"],
                        "onset_date": onset,
                        "severe_date": severe,
                        "warning": "本节仅用于实验审计，不得在面向用户的报告中作为病因诊断。",
                    },
                    "provenance": {
                        "protocol": "protocol_v2_clean / controlled_confirmation_v2",
                        "source_files": [
                            {"path": str(path.relative_to(ROOT)), "sha256": file_hash(path)}
                            for path in source_paths
                        ],
                        "evidence_generator": "scripts/physcrop_agent_v2_core.py",
                    },
                }
                evidence["evidence_hash"] = canonical_hash(evidence)
                path = EVIDENCE_DIR / f"{event_key}.json"
                write_json(path, evidence)
                connection.execute(
                    """INSERT INTO events(event_key,evidence_path,evidence_hash,risk_level,detected,updated_at,
                                          payload_uri,payload_sha256,source_type,schema_version)
                       VALUES(?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(event_key) DO UPDATE SET
                       evidence_path=excluded.evidence_path,evidence_hash=excluded.evidence_hash,
                       risk_level=excluded.risk_level,detected=excluded.detected,updated_at=excluded.updated_at,
                       payload_uri=excluded.payload_uri,payload_sha256=excluded.payload_sha256,
                       source_type=excluded.source_type,schema_version=excluded.schema_version""",
                    (
                        event_key, str(path), evidence["evidence_hash"], risk_level, int(detected), utc_now(),
                        f"event_evidence/{path.name}", file_hash(path), "anda_controlled_v2",
                        evidence["schema_version"],
                    ),
                )
                timeline_exists = connection.execute(
                    "SELECT 1 FROM event_timeline WHERE event_key=? LIMIT 1", (event_key,)
                ).fetchone()
                if timeline_exists is None:
                    connection.execute(
                        """INSERT INTO event_timeline
                           (event_key,actor,event_type,title,detail,created_at) VALUES(?,?,?,?,?,?)""",
                        (
                            event_key, "PhysCrop-F-v5", "detection", "模型完成事件评估",
                            f"风险等级：{RISK_LABELS[risk_level]}；受控演示数据。", utc_now(),
                        ),
                    )
                output_paths.append(path)
        connection.commit()
    finally:
        connection.close()
    return output_paths


def load_knowledge() -> dict[str, Any]:
    knowledge = read_json(KB_PATH)
    if knowledge.get("review_status") != "approved_for_internal_pilot":
        raise ValueError("Agronomy knowledge base is not approved for use")
    return knowledge


def retrieve_knowledge(evidence: dict[str, Any], knowledge: dict[str, Any]) -> list[dict[str, Any]]:
    features = evidence["evidence_summary"]["peak_spatial_features"]
    percentiles = evidence["evidence_summary"]["peak_feature_percentiles_within_seed"]
    canopy_strength = max(
        as_float(percentiles.get("ndvi_area_005"), 0.0),
        as_float(percentiles.get("ndvi_lcc_003"), 0.0),
        as_float(percentiles.get("ndvi_top_drop"), 0.0),
    )
    water_strength = max(
        as_float(percentiles.get("ndmi_area_003"), 0.0),
        as_float(percentiles.get("ndmi_lcc_002"), 0.0),
        as_float(percentiles.get("ndmi_top_drop"), 0.0),
    )
    tags: set[str] = set()
    if canopy_strength >= 60 or as_float(features.get("ndvi_area_005"), 0) >= 0.10:
        tags.add("canopy_signal")
    if water_strength >= 60 or as_float(features.get("ndmi_area_003"), 0) >= 0.10:
        tags.add("water_signal")
    if evidence["data_quality"]["status"] != "complete":
        tags.add("data_quality")

    source_index = {item["source_id"]: item for item in knowledge["sources"]}
    candidates = []
    for entry in knowledge["entries"]:
        trigger = set(entry["trigger_tags"])
        if trigger <= tags:
            expanded = dict(entry)
            expanded["sources"] = [source_index[source_id] for source_id in entry["source_ids"]]
            expanded["match_tags"] = sorted(trigger)
            candidates.append(expanded)
    candidates.sort(key=lambda item: (-len(item["trigger_tags"]), item["knowledge_id"]))
    return candidates[:3]


def _allowed_action_inventory(knowledge_entries: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    low = {action for entry in knowledge_entries for action in entry["low_risk_actions"]}
    conditional = {action for entry in knowledge_entries for action in entry["conditional_actions"]}
    return low, conditional


def _longest_true_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _trajectory_assessment(evidence: dict[str, Any]) -> dict[str, Any]:
    trajectory = evidence["trajectory"]
    above = [point["normalized_risk"] >= 1.0 for point in trajectory]
    above_points = [point for point, is_above in zip(trajectory, above) if is_above]
    peak = max(trajectory, key=lambda point: point["normalized_risk"])
    latest = trajectory[-1]
    first = trajectory[0]
    longest = _longest_true_run(above)
    first_above = above_points[0]["date"] if above_points else None
    after_peak = trajectory[trajectory.index(peak):]
    if len(after_peak) <= 1:
        post_peak = "峰值位于当前序列末端，尚无后续观测"
    elif latest["normalized_risk"] < peak["normalized_risk"] * 0.8:
        post_peak = "峰值后风险有所回落，但仍需结合阈值和现场复查"
    elif latest["normalized_risk"] >= 1.0:
        post_peak = "峰值后风险仍处于工作点以上"
    else:
        post_peak = "峰值后风险回到工作点以下"
    narrative = (
        f"共分析 {len(trajectory)} 个有效观测，其中 {len(above_points)} 个达到或超过工作点，"
        f"最长连续 {longest} 个观测。峰值为 {peak['normalized_risk']:.2f}，出现在 {peak['date']}；"
        f"{post_peak}。该趋势用于安排复查，不用于判断具体病因。"
    )
    return {
        "observation_count": len(trajectory),
        "above_threshold_count": len(above_points),
        "longest_consecutive_above": longest,
        "first_above_threshold_date": first_above,
        "first_observation_date": first["date"],
        "first_normalized_risk": first["normalized_risk"],
        "peak_date": peak["date"],
        "peak_normalized_risk": peak["normalized_risk"],
        "peak_model_votes": peak["model_votes"],
        "latest_date": latest["date"],
        "latest_normalized_risk": latest["normalized_risk"],
        "post_peak_pattern": post_peak,
        "narrative": narrative,
        "source_paths": ["trajectory", "model_assessment"],
    }


def _spatial_assessment(evidence: dict[str, Any]) -> dict[str, Any]:
    values = evidence["evidence_summary"]["peak_spatial_features"]
    percentiles = evidence["evidence_summary"]["peak_feature_percentiles_within_seed"]
    signals = []
    for key, (label, meaning) in FEATURE_GUIDE.items():
        if key not in values or key not in percentiles:
            continue
        signals.append({
            "feature_key": key,
            "label": label,
            "raw_value": values[key],
            "percentile": percentiles[key],
            "meaning": meaning,
            "source_paths": [
                f"evidence_summary.peak_spatial_features.{key}",
                f"evidence_summary.peak_feature_percentiles_within_seed.{key}",
            ],
        })
    signals.sort(key=lambda item: (-item["percentile"], item["feature_key"]))
    top = signals[:4]
    if top:
        summary = "、".join(f"{item['label']} P{item['percentile']:.0f}" for item in top[:3])
    else:
        summary = "无可用空间特征"
    return {
        "top_signals": top,
        "narrative": (
            f"峰值日期的主要空间证据为：{summary}。百分位仅表示相对同一测试种子样本的排序，"
            "这些特征描述变化范围和空间一致性，不能识别具体病因。"
        ),
        "interpretation_boundary": evidence["evidence_summary"]["interpretation_boundary"],
    }


def _candidate_questions(knowledge_id: str) -> list[str]:
    mapping = {
        "KB-CANOPY-CHECK-V1": [
            "风险区是否存在倒伏、缺苗、叶色改变、斑点或虫体？",
            "异常边界是否与机械通行、田块边界或农事操作一致？",
        ],
        "KB-WATER-BALANCE-V1": [
            "风险区与对照区的根层土壤水分是否存在稳定差异？",
            "近期是否出现灌溉不均、积水、排水不畅或持续高温少雨？",
        ],
        "KB-MIXED-TRIAGE-V1": [
            "冠层症状是否与地势、灌排方向或土壤水分变化共同出现？",
            "下一次复查时变化范围是否继续扩大或保持空间一致？",
        ],
    }
    return mapping.get(knowledge_id, ["补充作物、生育期和现场记录后重新评估。"])


def deterministic_report(evidence: dict[str, Any], knowledge: dict[str, Any]) -> dict[str, Any]:
    if evidence.get("schema_version") == "physcrop_risk_alert_evidence_v1":
        from physcrop_agent_v2_risk_report import deterministic_risk_report
        return deterministic_risk_report(evidence, knowledge)
    assessment = evidence["model_assessment"]
    identity = evidence["event_identity"]
    trajectory_assessment = _trajectory_assessment(evidence)
    spatial_assessment = _spatial_assessment(evidence)
    entries = retrieve_knowledge(evidence, knowledge)
    low_actions, conditional_actions = _allowed_action_inventory(entries)
    event_key = evidence["event_key"]
    risk_level = assessment["risk_level"]
    risk_label = assessment["risk_label_zh"]
    ordered_low = sorted(low_actions)
    if risk_level == "monitor":
        selected_low = [
            action for action in ordered_low
            if any(token in action for token in ("下一次有效卫星", "证据不足", "核对影像日期"))
        ][:3]
        selected_conditional: list[str] = []
    elif risk_level == "watch":
        selected_low = [
            action for action in ordered_low
            if any(token in action for token in ("拍摄", "土壤", "降水", "下一次有效卫星", "核对影像"))
        ][:4]
        selected_conditional = []
    else:
        selected_low = ordered_low[:7]
        selected_conditional = sorted(conditional_actions)
    detected_text = (
        f"模型在 {assessment['first_sustained_alert_date']} 首次形成连续两次告警"
        if assessment["sustained_alert"]
        else "模型尚未形成连续两次告警"
    )
    headline = f"{risk_label}：{detected_text}，需结合现场信息判断原因。"
    evidence_items = [
        {
            "title": "事件级判定",
            "field_path": "model_assessment.sustained_alert",
            "value": assessment["sustained_alert"],
            "interpretation": "连续两次观测达到5% FPR工作点，才计为持续事件告警。",
            "limitation": "该工作点来自受控测试协议，不等同于病因或损失程度。",
        },
        {
            "title": "峰值风险",
            "field_path": "model_assessment.peak_normalized_risk",
            "value": assessment["peak_normalized_risk"],
            "interpretation": "数值为模型分数除以对应模型种子的验证集阈值；1.0表示达到工作点。",
            "limitation": "归一化风险不是发生概率。",
        },
        {
            "title": "峰值日期",
            "field_path": "model_assessment.peak_date",
            "value": assessment["peak_date"],
            "interpretation": "该日期是当前观测序列中的最高归一化风险日期。",
            "limitation": "受卫星重访与有效观测日期限制。",
        },
    ]
    if assessment["lead_to_severe_days"] is not None:
        evidence_items.append({
            "title": "受控实验提前量",
            "field_path": "model_assessment.lead_to_severe_days",
            "value": assessment["lead_to_severe_days"],
            "interpretation": "首次持续告警日至受控 severe date 的天数。",
            "limitation": "仅为受控实验审计指标，真实部署中没有 severe 真值时不显示为业务结论。",
        })

    candidate_issues = [
        {
            "knowledge_id": entry["knowledge_id"],
            "title": entry["title"],
            "status": "待现场排查",
            "explanation": entry["candidate_issue"],
            "basis": entry["match_tags"],
            "confidence": "证据有限",
            "supporting_evidence": [
                item["label"] for item in spatial_assessment["top_signals"]
                if (
                    ("canopy_signal" in entry["match_tags"] and item["feature_key"].startswith("ndvi"))
                    or ("water_signal" in entry["match_tags"] and item["feature_key"].startswith("ndmi"))
                    or (len(entry["match_tags"]) > 1 and item["feature_key"].startswith("joint"))
                )
            ][:3],
            "missing_information": [
                "真实作物类型和生育期",
                "风险区与对照区的现场症状照片",
                "土壤水分、积排水和近期农事记录",
                "与事件日期对齐的原始天气观测",
            ],
            "field_questions": _candidate_questions(entry["knowledge_id"]),
        }
        for entry in entries
        if entry["knowledge_id"] != "KB-DATA-QUALITY-V1"
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
        "report_id": f"RPT-{event_key}-{ulid()}",
        "event_key": event_key,
        "language": "zh-CN",
        "status": "needs_review",
        "generated_at": utc_now(),
        "risk": {
            "level": risk_level,
            "label": risk_label,
            "decision_source": "deterministic_rule",
            "not_a_diagnosis": True,
        },
        "farmer_summary": {
            "headline": headline,
            "what_we_observed": f"{MODEL_NAME} 在该地块事件的峰值风险为 {assessment['peak_normalized_risk']:.2f} 倍阈值，峰值日期为 {assessment['peak_date']}。",
            "what_to_do_now": selected_low[:4],
            "important_note": "模型只能提示异常变化，不能仅凭遥感确定缺水、病虫害、缺肥或产量损失。",
        },
        "professional_analysis": {
            "conclusion": headline,
            "narrative": (
                f"当前风险等级为{risk_label}。峰值归一化风险为"
                f"{assessment['peak_normalized_risk']:.2f}，出现在 {assessment['peak_date']}；"
                "该数值表示相对验证阈值的比值，不是异常发生概率。候选原因仍需现场排查。"
            ),
            "event_overview": {
                "region": identity["region"],
                "year": identity["year"],
                "patch_id": identity["patch_id"],
                "crop_type": identity["crop_type"],
                "analysis_mode": evidence["mode"],
                "operating_point": assessment["operating_point"],
                "model_seed_count": evidence["data_quality"]["model_seed_count"],
                "observation_count": evidence["data_quality"]["observation_count"],
            },
            "trajectory_assessment": trajectory_assessment,
            "spatial_assessment": spatial_assessment,
            "evidence_items": evidence_items,
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
                f"当前为{risk_label}，处置目标是先用低风险现场检查补齐信息，再由人工决定是否升级。"
                "系统不会自动触发灌溉、施肥、施药或植保操作。"
            ),
            "within_24_hours": [
                action for action in selected_low
                if not any(token in action for token in ("下一次", "24至72", "复查"))
            ][:5],
            "within_24_to_72_hours": [
                action for action in selected_low
                if any(token in action for token in ("下一次", "24至72", "复查"))
            ],
            "after_confirmation": selected_conditional,
            "do_not_do": [
                "不要把风险比值解释为病害概率、缺水概率或减产比例。",
                "不要在缺少现场确认时直接施药、追肥或改变灌溉计划。",
                "不要把受控注入类型作为真实田间病因。",
            ],
        },
        "field_checklist": [
            {
                "item": "确认作物类型和生育期", "required": True,
                "method": "核对种植档案或向地块负责人确认。",
                "record": "作物、品种、生育期、播种日期。",
            },
            {
                "item": "风险区与正常区对照拍照", "required": True,
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
            "recommended_window": "24至72小时或下一次有效卫星观测后",
            "compare": ["归一化风险是否继续高于1.0", "风险空间范围是否扩大", "现场症状是否持续或加重"],
            "close_condition": "后续风险回到阈值以下、现场未见异常且数据质量问题得到解释，可降级为常规监测。",
            "escalate_condition": "连续告警持续、风险范围扩大或现场发现明确症状时，升级给农艺师。",
        },
        "escalation": {
            "rule": "持续告警、风险范围扩大、现场出现明显萎蔫/病斑/积水，或连续复查仍无法解释时，交由农艺师复核。",
            "human_approval_required": True,
        },
        "knowledge_citations": citations,
        "limitations": evidence["data_quality"]["limitations"],
        "provenance": {
            "evidence_hash": evidence["evidence_hash"],
            "knowledge_version": knowledge["version"],
            "generator": "deterministic_fallback",
            "model": MODEL_NAME,
        },
        "review": {"decision": "pending", "reviewer": None, "comment": None},
    }
    return report


def _get_path(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for segment in path.split("."):
        current = current[segment]
    return current


def validate_report(report: dict[str, Any], evidence: dict[str, Any],
                    knowledge: dict[str, Any]) -> dict[str, Any]:
    if evidence.get("schema_version") == "physcrop_risk_alert_evidence_v1":
        from physcrop_agent_v2_risk_report import validate_risk_report
        return validate_risk_report(report, evidence, knowledge)
    errors: list[str] = []
    warnings: list[str] = []
    required = {
        "schema_version", "event_key", "language", "status", "risk", "farmer_summary",
        "professional_analysis", "actions", "action_plan", "field_checklist", "recheck_plan", "escalation",
        "knowledge_citations", "limitations", "provenance", "review",
    }
    missing = sorted(required - set(report))
    if missing:
        errors.append(f"缺少必需字段: {', '.join(missing)}")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        errors.append("报告 schema_version 不匹配")
    if report.get("event_key") != evidence["event_key"]:
        errors.append("event_key 与证据不一致")
    if report.get("language") != "zh-CN":
        errors.append("报告必须输出 zh-CN")
    if report.get("risk", {}).get("level") != evidence["model_assessment"]["risk_level"]:
        errors.append("LLM 修改了规则确定的风险等级")
    if report.get("risk", {}).get("decision_source") != "deterministic_rule":
        errors.append("风险等级必须标记为 deterministic_rule")
    if report.get("actions", {}).get("automatically_executed"):
        errors.append("系统禁止自动执行农业处置")

    entries = retrieve_knowledge(evidence, knowledge)
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
    if report.get("risk", {}).get("level") in {"monitor", "watch"} and action_plan.get("after_confirmation"):
        errors.append("低风险事件不应包含确认后处置")

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

    assessment = evidence["model_assessment"]
    headline = report.get("farmer_summary", {}).get("headline", "")
    conclusion = report.get("professional_analysis", {}).get("conclusion", "")
    narrative = report.get("professional_analysis", {}).get("narrative", "")
    if assessment["risk_label_zh"] not in headline or assessment["risk_label_zh"] not in conclusion:
        errors.append("摘要或专业结论缺少规则确定的风险等级")
    if assessment["sustained_alert"]:
        alert_date = assessment["first_sustained_alert_date"]
        if alert_date not in headline or alert_date not in conclusion:
            errors.append("持续告警摘要缺少首次告警日期")
    required_narrative_terms = [
        assessment["risk_label_zh"],
        f"{assessment['peak_normalized_risk']:.2f}",
        assessment["peak_date"],
    ]
    if any(term not in narrative for term in required_narrative_terms):
        errors.append("专业叙事缺少风险等级、峰值或峰值日期")
    if "不是" not in narrative or "概率" not in narrative:
        errors.append("专业叙事没有说明归一化风险不是概率")
    if "排查" not in narrative and "核验" not in narrative:
        errors.append("专业叙事缺少现场排查边界")

    expected_trajectory = _trajectory_assessment(evidence)
    supplied_trajectory = report.get("professional_analysis", {}).get("trajectory_assessment", {})
    for key, expected in expected_trajectory.items():
        if key != "narrative" and supplied_trajectory.get(key) != expected:
            errors.append(f"时间轨迹统计与证据不一致: {key}")
    trajectory_narrative = supplied_trajectory.get("narrative", "")
    trajectory_terms = [
        str(expected_trajectory["observation_count"]),
        str(expected_trajectory["above_threshold_count"]),
        str(expected_trajectory["longest_consecutive_above"]),
        f"{expected_trajectory['peak_normalized_risk']:.2f}",
        expected_trajectory["peak_date"],
    ]
    if any(term not in trajectory_narrative for term in trajectory_terms):
        errors.append("时间轨迹叙事缺少锁定统计量")
    if "病因" not in trajectory_narrative or not any(term in trajectory_narrative for term in ("不用于", "不能")):
        errors.append("时间轨迹叙事缺少非诊断边界")

    expected_spatial = _spatial_assessment(evidence)
    supplied_spatial = report.get("professional_analysis", {}).get("spatial_assessment", {})
    if supplied_spatial.get("top_signals") != expected_spatial["top_signals"]:
        errors.append("空间证据排名与事实源不一致")
    if supplied_spatial.get("interpretation_boundary") != expected_spatial["interpretation_boundary"]:
        errors.append("空间证据解释边界被修改")
    spatial_narrative = supplied_spatial.get("narrative", "")
    for signal in expected_spatial["top_signals"][:2]:
        if signal["label"] not in spatial_narrative or f"P{signal['percentile']:.0f}" not in spatial_narrative:
            errors.append(f"空间叙事缺少锁定证据: {signal['feature_key']}")
    if "病因" not in spatial_narrative or not any(term in spatial_narrative for term in ("不能", "不识别")):
        errors.append("空间叙事缺少非诊断边界")

    action_rationale = action_plan.get("rationale", "")
    if assessment["risk_label_zh"] not in action_rationale or "现场" not in action_rationale or "人工" not in action_rationale:
        errors.append("行动理由缺少风险等级、现场核验或人工决策要求")

    content = json.dumps(report, ensure_ascii=False)
    forbidden_patterns = [
        r"(已确诊|确诊为|模型确诊|可以确诊|能够确诊)",
        r"可以确定.{0,8}(缺水|涝害|病害|虫害|缺肥)",
        r"亩用.{0,10}(克|毫升|公斤)", r"喷施.{0,20}(倍液|剂量)", r"保证增产",
        r"模型判断病因", r"注入类型表明",
    ]
    for pattern in forbidden_patterns:
        if re.search(pattern, content):
            errors.append(f"命中高风险或越界表达: {pattern}")
    if evidence["audit_context"]["injected_stress_type"] in content:
        errors.append("面向用户的报告泄露了受控注入类型")
    if not report.get("escalation", {}).get("human_approval_required"):
        errors.append("缺少人工批准门槛")
    if not report.get("risk", {}).get("not_a_diagnosis"):
        errors.append("缺少非诊断声明")
    if evidence["event_identity"]["crop_type"] == "unknown":
        warnings.append("作物类型未知，条件性农艺建议不得直接执行")
    return {"passes": not errors, "errors": errors, "warnings": warnings}


def build_llm_messages(evidence: dict[str, Any], knowledge: dict[str, Any],
                       draft: dict[str, Any]) -> list[dict[str, str]]:
    if evidence.get("schema_version") == "physcrop_risk_alert_evidence_v1":
        from physcrop_agent_v2_risk_report import build_risk_llm_messages
        return build_risk_llm_messages(evidence, knowledge, draft)
    entries = retrieve_knowledge(evidence, knowledge)
    compact_evidence = {
        "event_key": evidence["event_key"],
        "event_identity": evidence["event_identity"],
        "model_assessment": evidence["model_assessment"],
        "evidence_summary": evidence["evidence_summary"],
        "data_quality": evidence["data_quality"],
        "evidence_hash": evidence["evidence_hash"],
    }
    system = """你是 PhysCrop 农业事件解释 Agent。你只能重写用户提供的 JSON 草稿中的中文解释，必须返回单个合法 JSON 对象。
硬约束：
1. 不得修改 event_key、风险等级、任何数值、日期、证据路径、行动清单、引用、溯源和审核状态。
2. 不得使用 controlled stress type 或受控真值推断真实病因。
3. 不得诊断具体病虫害、缺水、涝害、缺素或产量损失；候选原因必须写成“待排查”。
4. 不得新增农药、肥料、灌溉剂量或任何知识库外行动。
5. 用清楚、克制、可执行的简体中文；区分农户摘要和专业分析，不重复堆砌同一句话。
6. 风险分数是阈值比，不是概率。最终发布必须由人工审核。
7. 时间和空间叙事中的锁定数字、日期、特征名称与百分位必须原样保留。"""
    user = {
        "task": (
            "不要复制 draft_report。返回一个精简 JSON，只包含 farmer_summary.headline、"
            "farmer_summary.important_note、professional_analysis.conclusion，以及四个新增字段："
            "professional_analysis.narrative_addition、professional_analysis.trajectory_assessment.narrative_addition、"
            "professional_analysis.spatial_assessment.narrative_addition、action_plan.rationale_addition。"
            "每个 addition 写2至3句、60至180字的补充解释，且不得出现任何阿拉伯数字：分别解释风险但不诊断、持续轨迹的业务意义、"
            "空间一致性与病因诊断的区别、先现场核验再人工决策的原因。不得新增数字、日期、病因或行动。"
        ),
        "evidence": compact_evidence,
        "retrieved_knowledge": entries,
        "draft_report": draft,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def parse_llm_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")
    return value


def merge_llm_narrative(draft: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Copy only prose fields from an LLM result into a rule-owned report."""
    merged = json.loads(json.dumps(draft, ensure_ascii=False))

    def append_explanation(base: str, proposed: Any, minimum: int, maximum: int) -> str:
        if not isinstance(proposed, str):
            return base
        proposed = proposed.strip()
        if not minimum <= len(proposed) <= maximum or proposed == base or re.search(r"\d", proposed):
            return base
        return f"{base}\n补充解释：{proposed}"

    def safe_rewrite(base: str, proposed: Any, minimum: int, maximum: int) -> str:
        if not isinstance(proposed, str):
            return base
        proposed = proposed.strip()
        proposed_numbers = set(re.findall(r"\d+(?:\.\d+)?", proposed))
        base_numbers = set(re.findall(r"\d+(?:\.\d+)?", base))
        if not minimum <= len(proposed) <= maximum or not proposed_numbers <= base_numbers:
            return base
        return proposed

    farmer_keys = ("headline", "important_note")
    for key in farmer_keys:
        value = candidate.get("farmer_summary", {}).get(key)
        merged["farmer_summary"][key] = safe_rewrite(draft["farmer_summary"][key], value, 4, 800)
    conclusion = candidate.get("professional_analysis", {}).get("conclusion")
    merged["professional_analysis"]["conclusion"] = safe_rewrite(
        draft["professional_analysis"]["conclusion"], conclusion, 4, 1200
    )
    narrative = candidate.get("professional_analysis", {}).get("narrative_addition")
    if narrative is None:
        narrative = candidate.get("professional_analysis", {}).get("narrative")
    merged["professional_analysis"]["narrative"] = append_explanation(
        draft["professional_analysis"]["narrative"], narrative, 20, 1200
    )
    trajectory_narrative = candidate.get("professional_analysis", {}).get("trajectory_assessment", {}).get("narrative_addition")
    if trajectory_narrative is None:
        trajectory_narrative = candidate.get("professional_analysis", {}).get("trajectory_assessment", {}).get("narrative")
    merged["professional_analysis"]["trajectory_assessment"]["narrative"] = append_explanation(
        draft["professional_analysis"]["trajectory_assessment"]["narrative"], trajectory_narrative, 30, 1200
    )
    spatial_narrative = candidate.get("professional_analysis", {}).get("spatial_assessment", {}).get("narrative_addition")
    if spatial_narrative is None:
        spatial_narrative = candidate.get("professional_analysis", {}).get("spatial_assessment", {}).get("narrative")
    merged["professional_analysis"]["spatial_assessment"]["narrative"] = append_explanation(
        draft["professional_analysis"]["spatial_assessment"]["narrative"], spatial_narrative, 30, 1200
    )
    action_rationale = candidate.get("action_plan", {}).get("rationale_addition")
    if action_rationale is None:
        action_rationale = candidate.get("action_plan", {}).get("rationale")
    merged["action_plan"]["rationale"] = append_explanation(
        draft["action_plan"]["rationale"], action_rationale, 20, 800
    )
    merged["provenance"]["generator"] = "llm_guarded"
    return merged


def persist_report(connection: sqlite3.Connection, report: dict[str, Any], validation: dict[str, Any],
                   generator: str, provider: str = "local", model: str = "rules-v1") -> Path:
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
            report["generated_at"], now, f"reports/{path.name}", file_hash(path),
            report.get("report_version", "1"), report.get("parent_report_id", ""),
            report.get("superseded_by", ""),
        ),
    )
    connection.commit()
    return path


def generate_deterministic_reports() -> list[Path]:
    knowledge = load_knowledge()
    connection = init_db()
    paths = []
    try:
        for evidence_path in sorted(EVIDENCE_DIR.glob("*.json")):
            evidence = read_json(evidence_path)
            report = deterministic_report(evidence, knowledge)
            validation = validate_report(report, evidence, knowledge)
            paths.append(persist_report(connection, report, validation, "deterministic_fallback"))
        audit(connection, "system", "generate_reports", "all_events", {"count": len(paths)})
    finally:
        connection.close()
    return paths


def load_event(event_key: str) -> dict[str, Any]:
    path = EVIDENCE_DIR / f"{event_key}.json"
    if not path.exists():
        raise FileNotFoundError(event_key)
    return read_json(path)


def load_report(event_key: str) -> dict[str, Any]:
    path = REPORT_DIR / f"{event_key}.json"
    if not path.exists():
        raise FileNotFoundError(event_key)
    return read_json(path)
