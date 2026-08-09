"""Read-only YieldSAT / PhysCrop-Risk adapter for the report agent v2 migration.

This module converts frozen PhysCrop-Risk artifacts into immutable
field-season evidence packages (``physcrop_risk_alert_evidence_v1``).
It never trains, tunes, or modifies the frozen results, and prospective
outputs never contain harvest outcomes.
"""

from __future__ import annotations

import csv
import hashlib
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
    utc_now,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "results" / "yieldsat_risk_v1"
FINAL = RISK / "final_audit"
POST = RISK / "post_audit_p0p1"
SPATIAL = RISK / "experiment_07_spatial_maps"
EARLY = RISK / "experiment_02_early_curve"
EVIDENCE_DIR = AGENT_ROOT / "risk_evidence"
RETRO_DIR = EVIDENCE_DIR / "retrospective"
MANIFEST_PATH = EVIDENCE_DIR / "risk_manifest.json"
INPUT_HASH_PATH = EVIDENCE_DIR / "risk_inputs.sha256"

SCHEMA_VERSION = "physcrop_risk_alert_evidence_v1"
ADAPTER_VERSION = "yieldsat_risk_adapter_v1.0"
ORGANIZATION_ID = "yieldsat-demo"
PERSISTENCE_REQUIRED = 2

INPUT_FILES = [
    FINAL / "final_predictions.csv",
    POST / "p0b_uniform_audit_predictions.csv",
    SPATIAL / "field_metrics.csv",
    EARLY / "milestones.json",
    EARLY / "target_reference.json",
    FINAL / "freeze_manifest.json",
    POST / "p0a_lead_times.csv",
]

OUTCOME_KEYS = {"target", "true_low", "covered_90", "covered_95", "harvest", "harvest_date", "lead_days"}
FIELD_ID_RE = re.compile(r"^([A-Za-z]+)_(DUP\d+_farm\d+)_field(\d+)_([a-z]+)_(\d{4})$")


class SchemaError(ValueError):
    pass


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_join(field_ids: list[str]) -> str:
    payload = "\n".join(sorted(field_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_inputs(strict: bool = True, manifest_path: Path | None = None) -> dict[str, str]:
    """Hash all frozen inputs and compare with the stored manifest."""
    current = {path.relative_to(ROOT).as_posix(): file_hash(path) for path in INPUT_FILES}
    path = manifest_path or INPUT_HASH_PATH
    if path.exists():
        stored = read_json(path)
        mismatched = [name for name, value in current.items() if stored.get(name) != value]
        if strict and mismatched:
            raise RuntimeError(f"Frozen risk input changed: {mismatched}")
    return current


def write_input_hash_manifest() -> dict[str, str]:
    hashes = verify_inputs(strict=False)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(INPUT_HASH_PATH, hashes)
    return hashes


def parse_field_id(field_id: str) -> dict[str, str]:
    match = FIELD_ID_RE.match(field_id)
    if not match:
        raise SchemaError(f"Unparseable YieldSAT field_id: {field_id}")
    country, farm, field_number, crop, year = match.groups()
    return {
        "country": country,
        "farm_id": f"{country}_{farm}",
        "field_number": field_number,
        "crop": crop,
        "year": year,
    }


def _load_frozen() -> dict[str, Any]:
    freeze = read_json(FINAL / "freeze_manifest.json")
    milestones = read_json(EARLY / "milestones.json")
    targets = read_json(EARLY / "target_reference.json")
    final_rows = read_csv(FINAL / "final_predictions.csv")
    p0b_rows = [row for row in read_csv(POST / "p0b_uniform_audit_predictions.csv") if row["model"] == "dual_tree"]
    spatial_rows = {row["field_id"]: row for row in read_csv(SPATIAL / "field_metrics.csv")}
    lead_rows = {(row["field_id"], int(row["look"])): row for row in read_csv(POST / "p0a_lead_times.csv")}

    final_by_field: dict[str, list[dict[str, Any]]] = {}
    for row in final_rows:
        final_by_field.setdefault(row["field_id"], []).append(row)
    for rows in final_by_field.values():
        rows.sort(key=lambda item: int(item["look"]))

    clean_watch: dict[tuple[str, int], int] = {}
    for row in p0b_rows:
        clean_watch[(row["field_id"], int(row["look"]))] = int(row["watch"])

    return {
        "freeze": freeze,
        "milestones": milestones,
        "targets": targets,
        "final_by_field": final_by_field,
        "clean_watch": clean_watch,
        "spatial": spatial_rows,
        "lead": lead_rows,
    }


def _look_history(field_id: str, bundle: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    rows = bundle["final_by_field"][field_id]
    meta = parse_field_id(field_id)
    crop = meta["crop"]
    milestones = bundle["milestones"][crop]["looks_gdd"]
    history: list[dict[str, Any]] = []
    for row in rows:
        look = int(row["look"])
        entry: dict[str, Any] = {
            "look": look,
            "look_type": "crop_specific_absolute_gdd",
            "gdd_cutoff": float(milestones[look - 1]),
            "prediction_t_ha": float(row["prediction"]),
            "low_yield_cutoff_t_ha": float(row["cutoff"]),
            "risk_score": float(row["score"]),
            "null_pvalue": float(row["pvalue"]),
            "alpha": float(row["alpha"]),
            "watch_original": int(row["watch"]),
            "action_original": int(row["action_alert"]),
            "watch_clean": int(bundle["clean_watch"].get((field_id, look), 0)),
            "calibration_method": row["method"],
            "baseline_t_ha": float(row["baseline"]),
            "upper_90": float(row["upper_90"]),
            "upper_95": float(row["upper_95"]),
        }
        lead = bundle["lead"].get((field_id, look))
        if lead:
            entry["cutoff_date"] = lead["last_acquisition"]
            if mode == "retrospective_audit":
                entry["harvest_date"] = lead["harvest"]
                entry["lead_days"] = int(lead["lead_days"])
        history.append(entry)
    for look in (1, 2, 3):
        if all(entry["look"] != look for entry in history):
            history.append({"look": look, "missing": True})
    history.sort(key=lambda item: item["look"])
    return history


def _current_row(history: list[dict[str, Any]]) -> dict[str, Any]:
    present = [entry for entry in history if not entry.get("missing")]
    if not present:
        raise SchemaError("Field has no usable look rows")
    return present[-1]


def _field_decisions(history: list[dict[str, Any]]) -> dict[str, Any]:
    present = [entry for entry in history if not entry.get("missing")]
    watches_original = [entry["watch_original"] for entry in present]
    watches_clean = [entry["watch_clean"] for entry in present]
    by_look_original = {entry["look"]: entry["watch_original"] for entry in present}
    by_look_clean = {entry["look"]: entry["watch_clean"] for entry in present}
    action_original = int(any(by_look_original.get(k, 0) and by_look_original.get(k - 1, 0) for k in (2, 3)))
    action_clean = int(any(by_look_clean.get(k, 0) and by_look_clean.get(k - 1, 0) for k in (2, 3)))
    return {
        "watch_original": int(any(watches_original)),
        "action_original": action_original,
        "watch_clean": int(any(watches_clean)),
        "action_clean": action_clean,
    }


def build_evidence(field_id: str, bundle: dict[str, Any], mode: str = "prospective_inference") -> dict[str, Any]:
    """Build one immutable evidence package for a field-season."""
    if mode not in {"prospective_inference", "retrospective_audit"}:
        raise SchemaError(f"Unknown mode: {mode}")
    if field_id not in bundle["final_by_field"]:
        raise SchemaError(f"Unknown field_id: {field_id}")
    meta = parse_field_id(field_id)
    freeze = bundle["freeze"]
    history = _look_history(field_id, bundle, mode)
    current = _current_row(history)
    decisions = _field_decisions(history)
    spatial = bundle["spatial"].get(field_id)
    look_count = sum(1 for entry in history if not entry.get("missing"))
    missing_looks = [look for entry in history if entry.get("missing") for look in (entry["look"],)]
    spatial_map_path = SPATIAL / "maps" / f"{field_id}.npz"

    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_id": f"ev-{field_id}",
        "mode": mode,
        "identity": {
            "organization_id": ORGANIZATION_ID,
            "farm_id": meta["farm_id"],
            "field_id": field_id,
            "field_number": meta["field_number"],
            "country": meta["country"],
            "crop": meta["crop"],
            "season": meta["year"],
        },
        "current_information_boundary": {
            "look": current["look"],
            "look_type": current["look_type"],
            "gdd_cutoff": current["gdd_cutoff"],
            "cutoff_date": current.get("cutoff_date"),
            "latest_used_acquisition_date": current.get("cutoff_date"),
            "preharvest_status": (
                "unknown_in_prospective_mode"
                if mode == "prospective_inference"
                else "verified_pre_harvest_from_frozen_audit"
            ),
        },
        "model_evidence": {
            "prediction_t_ha": current["prediction_t_ha"],
            "training_defined_low_yield_cutoff_t_ha": current["low_yield_cutoff_t_ha"],
            "risk_score": current["risk_score"],
            "null_pvalue": current["null_pvalue"],
            "look_alpha": current["alpha"],
            "watch": decisions["watch_clean"],
            "watch_original": decisions["watch_original"],
            "action": decisions["action_clean"],
            "action_original": decisions["action_original"],
            "persistence_required": PERSISTENCE_REQUIRED,
            "decision_policy": "post_audit_clean_null",
            "frozen_policy": "original_frozen",
        },
        "look_history": history,
        "uncertainty": {
            "method": "one_sided_split_conformal",
            "upper_90": current["upper_90"],
            "upper_95": current["upper_95"],
            "interpretation": "upper prediction evidence, not event probability",
        },
        "spatial_evidence": {
            "available": spatial is not None,
            "map_asset_id": f"maps/{field_id}.npz" if spatial_map_path.exists() else None,
            "pixels": int(spatial["pixels"]) if spatial else None,
            "rmse_spatial_t_ha": float(spatial["rmse_spatial"]) if spatial else None,
            "r2_spatial": float(spatial["r2_spatial"]) if spatial else None,
            "within_field_rho": float(spatial["rho_spatial"]) if spatial else None,
            "bottom20_auroc": float(spatial["bottom20_auroc"]) if spatial else None,
            "interpretation_boundary": "relative within-field evidence; not a causal or disease map",
        },
        "data_quality": {
            "frozen_results_only": True,
            "look_count": look_count,
            "missing_looks": missing_looks,
            "spatial_available": spatial is not None,
            "per_look_observation_counts": "not_available_in_frozen_csv",
            "limitations": [
                "证据由冻结审计结果重建；不暴露逐观测期的观测数量。",
                "p 值是与参考分布比较的越界证据，不是低产概率。",
                "空间指标是用于定位的田块级证据，不是诊断依据。",
            ],
        },
        "provenance": {
            "model_hash_look1": freeze["hashes"].get("results/yieldsat_risk_v1/experiment_02_early_curve/model_look1.joblib"),
            "model_hash_look2": freeze["hashes"].get("results/yieldsat_risk_v1/experiment_02_early_curve/model_look2.joblib"),
            "model_hash_look3": freeze["hashes"].get("results/yieldsat_risk_v1/experiment_02_early_curve/model_look3.joblib"),
            "feature_hash": freeze["hashes"].get("results/yieldsat_risk_v1/cache/audit_features_blind.npz"),
            "protocol_hash": freeze["hashes"].get("results/yieldsat_risk_v1/MASTER_PROTOCOL.md"),
            "freeze_manifest_sha256": freeze.get("freeze_manifest_sha256") or file_hash(FINAL / "freeze_manifest.json"),
            "adapter_version": ADAPTER_VERSION,
            "decision_policy_version": "original_frozen|post_audit_clean_null",
        },
    }
    if mode == "retrospective_audit":
        current_outcome = bundle["final_by_field"][field_id][-1]
        evidence["retrospective_block"] = {
            "sealed_for_audit": True,
            "target_t_ha": float(current_outcome["target"]),
            "true_low": int(current_outcome["true_low"]),
            "covered_90": int(current_outcome["covered_90"]),
            "covered_95": int(current_outcome["covered_95"]),
            "harvest_date": current.get("harvest_date"),
            "lead_days": current.get("lead_days"),
            "not_for_prospective_use": True,
        }
    evidence["evidence_hash"] = canonical_hash(evidence)
    return evidence


def _assert_no_outcome_keys(evidence: dict[str, Any]) -> None:
    text = json.dumps(evidence, ensure_ascii=False)
    for key in OUTCOME_KEYS:
        if f'"{key}"' in text:
            raise SchemaError(f"Prospective evidence leaked outcome key: {key}")


def build_all_evidence(mode: str = "retrospective_audit") -> dict[str, Any]:
    """Build prospective + retrospective evidence for all 322 audit fields."""
    bundle = _load_frozen()
    field_ids = sorted(bundle["final_by_field"].keys())
    freeze = bundle["freeze"]
    expected_field_hash = freeze["field_ids_sha256"]
    actual_field_hash = _sha256_join(field_ids)
    if actual_field_hash != expected_field_hash:
        raise SchemaError("Audit field-id hash mismatch with freeze manifest")
    write_input_hash_manifest()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    RETRO_DIR.mkdir(parents=True, exist_ok=True)
    look_row_count = 0
    decisions = {"watch_original": 0, "action_original": 0, "watch_clean": 0, "action_clean": 0}
    spatial_available = 0
    for field_id in field_ids:
        prospective = build_evidence(field_id, bundle, "prospective_inference")
        _assert_no_outcome_keys(prospective)
        retrospective = build_evidence(field_id, bundle, "retrospective_audit")
        write_json(EVIDENCE_DIR / f"{field_id}.json", prospective)
        write_json(RETRO_DIR / f"{field_id}.json", retrospective)
        look_row_count += len([e for e in prospective["look_history"] if not e.get("missing")])
        decisions["watch_original"] += int(prospective["model_evidence"]["watch_original"])
        decisions["action_original"] += int(prospective["model_evidence"]["action_original"])
        decisions["watch_clean"] += int(prospective["model_evidence"]["watch"])
        decisions["action_clean"] += int(prospective["model_evidence"]["action"])
        spatial_available += int(prospective["spatial_evidence"]["available"])
    manifest = {
        "schema_version": "physcrop_risk_evidence_manifest_v1",
        "adapter_version": ADAPTER_VERSION,
        "generated_at": utc_now(),
        "field_count": len(field_ids),
        "field_ids_sha256": actual_field_hash,
        "look_row_count": look_row_count,
        "decision_counts": decisions,
        "spatial_available_fields": spatial_available,
        "mode": "prospective_inference + retrospective_audit",
        "outcome_leakage_check": "passed_for_all_prospective_files",
    }
    write_json(MANIFEST_PATH, manifest)
    return manifest


def _risk_level_for_event(evidence: dict[str, Any]) -> str:
    model = evidence["model_evidence"]
    if model["action"]:
        return "priority"
    if model["watch"]:
        return "watch"
    return "monitor"


def seed_risk_events() -> dict[str, Any]:
    """Insert the 322 risk fields into the v2 SQLite workspace."""
    bundle = _load_frozen()
    field_ids = sorted(bundle["final_by_field"].keys())
    connection = init_db()
    now = utc_now()
    try:
        connection.execute(
            "INSERT OR IGNORE INTO organizations(organization_id,name,created_at) VALUES(?,?,?)",
            (ORGANIZATION_ID, "YieldSAT 商业演示组织", now),
        )
        farm_ids: set[str] = set()
        field_rows: list[tuple[str, str, str, str, str, int, int, str]] = []
        event_rows: list[tuple[str, str, str, str, int, str, str, str, str, str, str, str, int, int, int, int, int, int, int, str]] = []
        for field_id in field_ids:
            prospective = read_json(EVIDENCE_DIR / f"{field_id}.json")
            meta = parse_field_id(field_id)
            farm_id = meta["farm_id"]
            farm_ids.add(farm_id)
            field_rows.append(
                (
                    field_id, farm_id, ORGANIZATION_ID, meta["country"], meta["crop"],
                    int(meta["year"]), int(prospective["spatial_evidence"]["available"]), now,
                )
            )
            model = prospective["model_evidence"]
            event_rows.append(
                (
                    field_id, f"risk_evidence/{field_id}.json", prospective["evidence_hash"],
                    _risk_level_for_event(prospective), int(model["action"]), now,
                    f"risk_evidence/{field_id}.json",
                    hashlib.sha256(Path(EVIDENCE_DIR / f"{field_id}.json").read_bytes()).hexdigest(),
                    "yieldsat_risk_v1", SCHEMA_VERSION,
                    meta["country"], meta["crop"], int(meta["year"]),
                    model["watch_original"], model["action_original"],
                    model["watch"], model["action"],
                    prospective["current_information_boundary"]["look"],
                )
            )
        for farm_id in sorted(farm_ids):
            connection.execute(
                "INSERT OR IGNORE INTO farms(farm_id,organization_id,name,country,created_at) VALUES(?,?,?,?,?)",
                (farm_id, ORGANIZATION_ID, farm_id, farm_id.split("_")[0], now),
            )
        connection.executemany(
            """INSERT OR REPLACE INTO fields
               (field_id,farm_id,organization_id,country,crop,year,spatial_available,inserted_at)
               VALUES(?,?,?,?,?,?,?,?)""",
            field_rows,
        )
        connection.executemany(
            """INSERT OR REPLACE INTO events
               (event_key,evidence_path,evidence_hash,risk_level,detected,updated_at,
                payload_uri,payload_sha256,source_type,schema_version,
                country,crop,season_year,watch_original,action_original,watch_clean,action_clean,current_look)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            event_rows,
        )
        connection.commit()
        audit(connection, "system", "seed_risk_events", "yieldsat_risk_v1", {"field_count": len(field_ids)})
    finally:
        connection.close()
    return {"field_count": len(field_ids), "farm_count": len(farm_ids), "status": "seeded"}


def main() -> None:
    manifest = build_all_evidence("retrospective_audit")
    seeded = seed_risk_events()
    print(json.dumps({"evidence": manifest, "database": seeded}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
