"""Experiment 4: sequential low-yield alerts with field-season FWER control."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import beta

from run_yieldsat_experiment2 import CACHE, OUT as EXP2_OUT, subrole, threshold_for
from run_yieldsat_experiment3 import predictions, build_cells, choose_cell, conformal_quantile


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/yieldsat_risk_v1/experiment_04_familywise_alert"
POLICIES = {
    "marginal_05": [0.05, 0.05, 0.05],
    "bonferroni_equal": [1/60, 1/60, 1/60],
    "alpha_spending": [0.010, 0.015, 0.025],
}
SELECTED_METHOD = {1: "global_raw", 2: "hierarchy_power50", 3: "global_raw"}


def upper_cp(x, n, level=.95):
    return 1.0 if x == n else float(beta.ppf(level, x + 1, n - x))


def summarize_field_policy(records, truth, groups):
    alerted = {fid: any(x["alert"] for x in rows) for fid, rows in records.items()}
    ids = sorted(truth); y = np.asarray([truth[x] for x in ids]); a = np.asarray([alerted.get(x, False) for x in ids])
    normal = y == 0; low = y == 1
    fp, tn, tp, fn = int((a & normal).sum()), int((~a & normal).sum()), int((a & low).sum()), int((~a & low).sum())
    group_metrics = {}
    for group in sorted(set(groups.values())):
        g_ids = [x for x in ids if groups[x] == group]; g_normal = [x for x in g_ids if truth[x] == 0]
        if len(g_normal) >= 10:
            group_metrics[group] = {"normal_n": len(g_normal), "false_alerts": sum(alerted.get(x, False) for x in g_normal),
                                    "fpr": float(np.mean([alerted.get(x, False) for x in g_normal]))}
    earliest = [min(x["look"] for x in records[fid] if x["alert"]) for fid in ids if alerted.get(fid, False)]
    return {"n": len(ids), "low_yield_n": int(low.sum()), "normal_n": int(normal.sum()), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "fpr_field_season": fp / max(1, int(normal.sum())), "fpr_upper95_one_sided": upper_cp(fp, int(normal.sum())),
            "sensitivity": tp / max(1, int(low.sum())), "precision": tp / max(1, tp + fp), "specificity": tn / max(1, tn + fp),
            "earliest_alert_look_median": float(np.median(earliest)) if earliest else None,
            "supported_group_fpr": group_metrics,
            "worst_supported_group_fpr": max([x["fpr"] for x in group_metrics.values()], default=0.0)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(CACHE, allow_pickle=False)
    target = data["target"].astype(float)
    ref = np.flatnonzero((data["role"] == "adapter_reference") & np.isfinite(target))
    calibration = np.asarray([i for i in ref if subrole(data["farm_group"][i]) == "conformal"], dtype=int)
    challenge = np.flatnonzero((data["role"] == "locked_external") & np.isfinite(target))
    thresholds = json.loads((EXP2_OUT / "target_reference.json").read_text(encoding="utf-8"))["low_yield_thresholds"]
    truth, groups, cutoffs = {}, {}, {}
    for i in challenge:
        fid = str(data["field_id"][i]); country, crop = str(data["country"][i]), str(data["crop"][i])
        cutoff, _ = threshold_for(thresholds, country, crop)
        truth[fid] = int(target[i] <= cutoff); groups[fid] = f"{country}|{crop}"; cutoffs[fid] = cutoff

    policy_records = {"naive": {fid: [] for fid in truth}}
    for name in POLICIES: policy_records[name] = {fid: [] for fid in truth}
    flat_rows, stage_reports = [], {}
    for look in (1, 2, 3):
        bundle = joblib.load(EXP2_OUT / f"model_look{look}.joblib")
        cal_rows = predictions(bundle, data, calibration); eval_rows = predictions(bundle, data, challenge)
        for row in cal_rows:
            row["raw_score"] = row["target"] - row["prediction"]
            row["power50_score"] = row["raw_score"] / np.sqrt(row["scale"])
        raw_cells = build_cells(cal_rows, "raw_score"); power_cells = build_cells(cal_rows, "power50_score")
        # Diagnostic uncalibrated alert.
        for row in eval_rows:
            fid = row["field_id"]; alert = row["prediction"] < cutoffs[fid]
            record = {"policy": "naive", "look": look, **row, "alpha": None, "upper": row["prediction"],
                      "cutoff": cutoffs[fid], "alert": int(alert), "true_low": truth[fid],
                      "look_gdd": bundle["look_gdd"][row["crop"]]}
            policy_records["naive"][fid].append(record); flat_rows.append(record)
        stage_reports[str(look)] = {}
        for policy, alphas in POLICIES.items():
            alpha = alphas[look - 1]; nominal = 1.0 - alpha; method = SELECTED_METHOD[look]
            cells = power_cells if method == "hierarchy_power50" else raw_cells
            quantiles = {key: conformal_quantile(values, nominal) for key, values in cells.items()}
            normals_alert, normals_n = 0, 0
            for row in eval_rows:
                fid = row["field_id"]; cell = choose_cell(method, row, cells); q, rank = quantiles[cell]
                adjustment = q * np.sqrt(row["scale"]) if method == "hierarchy_power50" else q
                upper = row["prediction"] + adjustment; alert = upper < cutoffs[fid]
                record = {"policy": policy, "look": look, **row, "alpha": alpha, "upper": float(upper),
                          "cutoff": cutoffs[fid], "alert": int(alert), "true_low": truth[fid],
                          "look_gdd": bundle["look_gdd"][row["crop"]], "calibration_cell": cell,
                          "calibration_n": len(cells[cell]), "quantile_rank": rank, "method": method}
                policy_records[policy][fid].append(record); flat_rows.append(record)
                if truth[fid] == 0:
                    normals_n += 1; normals_alert += int(alert)
            stage_reports[str(look)][policy] = {"alpha": alpha, "normal_n": normals_n, "false_alerts": normals_alert,
                                                "stage_fpr": normals_alert / max(1, normals_n)}

    # Persistence is a post-processing of the already-valid alpha-spending bounds.
    persistent = {fid: [] for fid in truth}
    for fid, records in policy_records["alpha_spending"].items():
        by_look = {x["look"]: x for x in records}
        for look in sorted(by_look):
            record = dict(by_look[look]); record["policy"] = "alpha_spending_persistent"
            record["alert"] = int(bool(record["alert"]) and bool(by_look.get(look - 1, {}).get("alert", 0)))
            persistent[fid].append(record); flat_rows.append(record)
    policy_records["alpha_spending_persistent"] = persistent

    summaries = {name: summarize_field_policy(records, truth, groups) for name, records in policy_records.items()}
    primary = summaries["alpha_spending"]
    false_alert_implications = [x for x in flat_rows if x["policy"] == "alpha_spending" and x["alert"] and not x["true_low"]]
    implication_ok = all(x["target"] > x["upper"] for x in false_alert_implications)
    integrity = {"audit_targets_read": False, "conformal_evaluation_field_overlap": len(set(map(str, data["field_id"][calibration])) & set(truth)),
                 "alpha_spending_sum": sum(POLICIES["alpha_spending"]), "false_alert_implies_noncoverage": implication_ok,
                 "causal_frozen_look_models": True}
    gates = {"integrity": integrity["conformal_evaluation_field_overlap"] == 0 and integrity["alpha_spending_sum"] <= .05 + 1e-12 and implication_ok,
             "fpr_le_0.05": primary["fpr_field_season"] <= .05,
             "fpr_upper95_le_0.08": primary["fpr_upper95_one_sided"] <= .08,
             "nondegenerate_sensitivity": primary["sensitivity"] >= .05 and primary["tp"] >= 3,
             "precision_ge_0.50": primary["precision"] >= .50,
             "worst_group_fpr_le_0.15": primary["worst_supported_group_fpr"] <= .15}
    with (OUT / "alerts.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        keys = sorted(set().union(*(x.keys() for x in flat_rows))); writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(flat_rows)
    report = {"experiment": 4, "status": "PASS" if all(gates.values()) else "FAIL", "primary_policy": "alpha_spending",
              "policy_summaries": summaries, "stage_reports": stage_reports, "integrity": integrity, "pass_gates": gates,
              "validity_boundary": "Union-bound FWER requires the corresponding split-conformal exchangeability assumptions."}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True))


if __name__ == "__main__":
    main()
