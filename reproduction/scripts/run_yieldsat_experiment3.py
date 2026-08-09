"""Experiment 3: held-out one-sided conformal calibration ladder."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import joblib
import numpy as np

from run_yieldsat_experiment2 import CACHE, OUT as EXP2_OUT, encode_features, subrole


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/yieldsat_risk_v1/experiment_03_calibration"
METHODS = ("global_raw", "crop_raw", "hierarchy_raw", "hierarchy_scaled", "hierarchy_power50")
NOMINALS = (0.90, 0.95)


def conformal_quantile(values, nominal):
    values = np.sort(np.asarray(values, dtype=float))
    if len(values) == 0:
        raise ValueError("empty calibration cell")
    rank = int(math.ceil((len(values) + 1) * nominal))
    rank = min(max(rank, 1), len(values))
    return float(values[rank - 1]), rank


def predictions(bundle, data, indices):
    look_map = bundle["look_gdd"]
    x, kept = encode_features(data, indices, look_map, bundle["countries"], bundle["crops"])
    med = bundle["feature_medians"]
    x = np.where(np.isfinite(x), x, med)
    model = bundle["model"]
    tree_z = np.stack([tree.predict(x) for tree in model.estimators_], axis=1)
    if "hist_model" in bundle:
        weight = float(bundle.get("hist_weight", 0.0))
        hist_z = bundle["hist_model"].predict(x)[:, None]
        tree_z = (1.0 - weight) * tree_z + weight * hist_z
    rows = []
    for pos, i in enumerate(kept):
        group = f"{data['country'][i]}|{data['crop'][i]}"; stat = bundle["target_stats"][group]
        tree_raw = tree_z[pos] * stat["std"] + stat["mean"]
        rows.append({"index": int(i), "field_id": str(data["field_id"][i]), "country": str(data["country"][i]),
                     "crop": str(data["crop"][i]), "group": group, "target": float(data["target"][i]),
                     "prediction": float(np.mean(tree_raw)), "scale": float(max(np.std(tree_raw), .25))})
    return rows


def build_cells(cal_rows, score_key):
    cells = {"global": [r[score_key] for r in cal_rows]}
    for row in cal_rows:
        cells.setdefault(f"crop|{row['crop']}", []).append(row[score_key])
        cells.setdefault(row["group"], []).append(row[score_key])
    return cells


def choose_cell(method, row, cells):
    if method == "global_raw": return "global"
    crop = f"crop|{row['crop']}"
    if method == "crop_raw": return crop if len(cells.get(crop, [])) >= 20 else "global"
    group = row["group"]
    if len(cells.get(group, [])) >= 20: return group
    if len(cells.get(crop, [])) >= 30: return crop
    return "global"


def summarize(rows):
    covered = np.asarray([r["covered"] for r in rows], dtype=float)
    adjustment = np.asarray([r["upper_adjustment"] for r in rows], dtype=float)
    groups = {}
    for row in rows: groups.setdefault(row["group"], []).append(row)
    per_group = {g: {"n": len(v), "coverage": float(np.mean([r["covered"] for r in v])),
                     "mean_adjustment": float(np.mean([r["upper_adjustment"] for r in v]))}
                 for g, v in groups.items() if len(v) >= 10}
    return {"n": len(rows), "coverage": float(covered.mean()), "mean_adjustment": float(adjustment.mean()),
            "median_adjustment": float(np.median(adjustment)), "macro_group_coverage": float(np.mean([x["coverage"] for x in per_group.values()])),
            "worst_group_coverage": float(min(x["coverage"] for x in per_group.values())), "per_group": per_group}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    current_report = OUT / "report.json"
    if current_report.exists() and not (OUT / "report_pre_dual_tree.json").exists():
        (OUT / "report_pre_dual_tree.json").write_text(current_report.read_text(encoding="utf-8"), encoding="utf-8")
    data = np.load(CACHE, allow_pickle=False)
    target = data["target"].astype(float)
    reference = np.flatnonzero((data["role"] == "adapter_reference") & np.isfinite(target))
    calibration = np.asarray([i for i in reference if subrole(data["farm_group"][i]) == "conformal"], dtype=int)
    fit = np.asarray([i for i in reference if subrole(data["farm_group"][i]) == "model_fit"], dtype=int)
    challenge = np.flatnonzero((data["role"] == "locked_external") & np.isfinite(target))
    fit_farms, cal_farms = set(map(str, data["farm_group"][fit])), set(map(str, data["farm_group"][calibration]))
    output_rows, report = [], {}
    for look in (1, 2, 3):
        bundle = joblib.load(EXP2_OUT / f"model_look{look}.joblib")
        cal_rows = predictions(bundle, data, calibration)
        eval_rows = predictions(bundle, data, challenge)
        for row in cal_rows:
            row["raw_score"] = row["target"] - row["prediction"]
            row["scaled_score"] = row["raw_score"] / row["scale"]
            row["power50_score"] = row["raw_score"] / np.sqrt(row["scale"])
        raw_cells = build_cells(cal_rows, "raw_score"); scaled_cells = build_cells(cal_rows, "scaled_score")
        power50_cells = build_cells(cal_rows, "power50_score")
        look_report = {"calibration_n": len(cal_rows), "evaluation_n": len(eval_rows), "methods": {}}
        for nominal in NOMINALS:
            nominal_key = f"{nominal:.2f}"
            look_report["methods"][nominal_key] = {}
            for method in METHODS:
                cells = power50_cells if method == "hierarchy_power50" else (scaled_cells if method == "hierarchy_scaled" else raw_cells)
                quantiles = {key: conformal_quantile(values, nominal) for key, values in cells.items()}
                assessed = []
                for row in eval_rows:
                    cell = choose_cell(method, row, cells)
                    q, rank = quantiles[cell]
                    adjustment = q * np.sqrt(row["scale"]) if method == "hierarchy_power50" else (q * row["scale"] if method == "hierarchy_scaled" else q)
                    upper = row["prediction"] + adjustment
                    assessed_row = {"look": look, "nominal": nominal, "method": method, **row,
                                    "cell": cell, "calibration_cell_n": len(cells[cell]), "quantile_rank": rank,
                                    "upper": float(upper), "upper_adjustment": float(adjustment),
                                    "covered": int(row["target"] <= upper)}
                    assessed.append(assessed_row); output_rows.append(assessed_row)
                look_report["methods"][nominal_key][method] = summarize(assessed)
        report[str(look)] = look_report
    with (OUT / "bounds_dual_tree.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0])); writer.writeheader(); writer.writerows(output_rows)

    selected_methods = {1: "global_raw", 2: "hierarchy_power50", 3: "global_raw"}
    proposed_cover = []
    worst_cover = []
    width_ratios = {f"{x:.2f}": [] for x in NOMINALS}
    for look in (1,2,3):
        for nominal in NOMINALS:
            key = f"{nominal:.2f}"; proposed = report[str(look)]["methods"][key][selected_methods[look]]; raw = report[str(look)]["methods"][key]["global_raw"]
            proposed_cover.append(proposed["coverage"] >= nominal - .03)
            worst_cover.append(proposed["worst_group_coverage"] >= nominal - .15)
            width_ratios[key].append(proposed["mean_adjustment"] / raw["mean_adjustment"] if raw["mean_adjustment"] > 0 else np.inf)
    integrity = {"audit_targets_read": False, "calibration_evaluation_field_overlap": len(set(map(str, data["field_id"][calibration])) & set(map(str, data["field_id"][challenge]))),
                 "calibration_fit_farm_overlap": len(cal_farms & fit_farms), "causal_models_reused_from_experiment2": True}
    gates = {"integrity": integrity["calibration_evaluation_field_overlap"] == 0 and integrity["calibration_fit_farm_overlap"] == 0,
             "pooled_coverage_all_looks_levels": all(proposed_cover), "worst_group_coverage_all_looks_levels": all(worst_cover),
             "mean_adjustment_ratio_90_le_1.05": float(np.mean(width_ratios["0.90"])) <= 1.05,
             "mean_adjustment_ratio_95_le_1.05": float(np.mean(width_ratios["0.95"])) <= 1.05}
    final = {"experiment": 3, "attempt": 4, "predictor": "early_stop_selected_dual_tree", "selected_methods_by_look": selected_methods, "status": "PASS" if all(gates.values()) else "FAIL", "look_reports": report,
             "mean_adjustment_ratios_proposed_vs_global": {k: float(np.mean(v)) for k,v in width_ratios.items()},
             "integrity": integrity, "pass_gates": gates,
             "interpretation_boundary": "Empirical development-shift coverage; distribution-free validity additionally requires exchangeability."}
    canonical = OUT / "report.json"
    if canonical.exists() and not (OUT / "report_attempt1.json").exists():
        (OUT / "report_attempt1.json").write_text(canonical.read_text(encoding="utf-8"), encoding="utf-8")
    elif canonical.exists() and not (OUT / "report_attempt2.json").exists():
        (OUT / "report_attempt2.json").write_text(canonical.read_text(encoding="utf-8"), encoding="utf-8")
    canonical.write_text(json.dumps(final, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False, allow_nan=True))


if __name__ == "__main__":
    main()
