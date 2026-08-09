"""Independently validate Experiment 10 predictions, metrics, and causality."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_yieldsat_experiment2 import CACHE, OUT as EXP2_OUT  # noqa: E402
from run_yieldsat_experiment5 import group_metrics, metrics  # noqa: E402


def main() -> None:
    out = ROOT / "results" / "yieldsat_risk_v1" / "experiment_10_vision_baselines"
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    with (out / "predictions.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    data = np.load(CACHE, allow_pickle=False)
    target = data["target"].astype(float)
    test = np.flatnonzero((data["role"] == "locked_external") & np.isfinite(target))
    bundle = joblib.load(EXP2_OUT / "model_look3.joblib")
    thresholds = json.loads((EXP2_OUT / "target_reference.json").read_text(encoding="utf-8"))["low_yield_thresholds"]
    checks: dict[str, bool] = {}
    checks["row_count_316"] = len(rows) == len(test) == 316
    checks["field_order"] = [row["field_id"] for row in rows] == data["field_id"][test].astype(str).tolist()
    checks["targets"] = bool(np.allclose([float(row["target"]) for row in rows], target[test], atol=1e-7))
    for method in ("agrifm_frozen_context", "convlstm_context"):
        prediction = np.asarray([float(row[method]) for row in rows])
        recomputed = metrics(target[test], prediction, test, data, thresholds)
        expected = report["models"][method]["overall"]
        checks[f"{method}_finite"] = bool(np.isfinite(prediction).all())
        checks[f"{method}_overall_metrics"] = all(
            abs(float(recomputed[key]) - float(expected[key])) <= 1e-10
            for key in ("rmse", "mae", "r2", "spearman_rho", "auroc", "auprc", "prevalence")
        )
        grouped = group_metrics(target[test], prediction, test, data)
        macro = float(np.mean([value["r2"] for value in grouped.values()]))
        worst = float(min(value["r2"] for value in grouped.values() if value["n"] >= 10))
        checks[f"{method}_group_metrics"] = (
            abs(macro - float(report["models"][method]["macro_group_r2"])) <= 1e-10
            and abs(worst - float(report["models"][method]["worst_group_r2_n_ge_10"])) <= 1e-10
        )
    image_cache = np.load(out / "look3_image_sequences_32.npz", allow_pickle=False)
    cutoffs = np.asarray([bundle["look_gdd"][str(crop)] for crop in data["crop"]], dtype=float)
    checks["all_cached_frames_causal"] = bool(np.all(image_cache["selected_gdd"] <= cutoffs[:, None] + 1e-6))
    checks["report_pass"] = report["status"] == "PASS"
    validation = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    (out / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))
    if validation["status"] != "PASS":
        raise RuntimeError("Experiment 10 validation failed")


if __name__ == "__main__":
    main()
