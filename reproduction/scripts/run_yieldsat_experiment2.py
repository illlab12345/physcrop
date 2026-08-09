"""Experiment 2: causal absolute-GDD early yield prediction curves."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, average_precision_score


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results/yieldsat_risk_v1/cache/development_sequences.npz"
OUT = ROOT / "results/yieldsat_risk_v1/experiment_02_early_curve"
LOOK_FRACTIONS = np.asarray([0.35, 0.50, 0.65])
INTERP_FRACTIONS = np.linspace(0.0, 1.0, 6)


def bucket(text):
    return int(hashlib.sha256(("yieldsat-risk-v1-role:" + str(text)).encode()).hexdigest()[:12], 16) % 100


def subrole(farm):
    value = bucket(farm)
    return "model_fit" if value < 70 else ("early_stop" if value < 85 else "conformal")


def metric(y, p):
    rho = spearmanr(y, p).statistic if len(y) > 2 else np.nan
    return {"n": int(len(y)), "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae": float(mean_absolute_error(y, p)), "r2": float(r2_score(y, p)) if len(y) > 1 else np.nan,
            "spearman_rho": float(rho) if np.isfinite(rho) else np.nan}


def encode_features(data, indices, look_by_crop, countries, crops):
    rows, kept = [], []
    for i in indices:
        crop = str(data["crop"][i]); look = look_by_crop[crop]
        length = int(data["lengths"][i]); seq = data["dynamic"][i, :length]
        gdd = seq[:, 17].astype(float)
        use = np.flatnonzero(np.isfinite(gdd) & (gdd <= look + 1e-6))
        if len(use) < 3:
            continue
        seq = seq[use]; gdd = gdd[use]
        # 12 S2 + 4 vegetation indices + valid fraction + cumulative precipitation.
        selected = np.r_[np.arange(17), 18]
        query = INTERP_FRACTIONS * look
        temporal = []
        for col in selected:
            values = seq[:, col].astype(float)
            finite = np.isfinite(values)
            if finite.sum() == 0:
                temporal.extend([np.nan] * len(query))
            elif finite.sum() == 1:
                temporal.extend([float(values[finite][0])] * len(query))
            else:
                temporal.extend(np.interp(query, gdd[finite], values[finite]).tolist())
        static = data["static"][i].astype(float).tolist()
        onehot = [float(str(data["country"][i]) == x) for x in countries]
        onehot += [float(crop == x) for x in crops]
        rows.append(temporal + static + onehot)
        kept.append(int(i))
    return np.asarray(rows, dtype=np.float32), np.asarray(kept, dtype=int)


def group_stats(data, fit):
    stats, thresholds = {}, {}
    y = data["target"].astype(float)
    for country in sorted(set(map(str, data["country"]))):
        for crop in sorted(set(map(str, data["crop"]))):
            idx = [i for i in fit if str(data["country"][i]) == country and str(data["crop"][i]) == crop]
            if not idx:
                continue
            vals = y[idx]
            stats[f"{country}|{crop}"] = {"n": len(idx), "mean": float(vals.mean()), "std": float(max(vals.std(), 0.25))}
            if len(idx) >= 30:
                thresholds[f"{country}|{crop}"] = {"level": "country_crop", "n": len(idx), "q20": float(np.quantile(vals, .2))}
    for crop in sorted(set(map(str, data["crop"]))):
        idx = [i for i in fit if str(data["crop"][i]) == crop]
        if len(idx) >= 50:
            thresholds[f"crop|{crop}"] = {"level": "crop", "n": len(idx), "q20": float(np.quantile(y[idx], .2))}
    thresholds["global"] = {"level": "global", "n": len(fit), "q20": float(np.quantile(y[fit], .2))}
    return stats, thresholds


def threshold_for(thresholds, country, crop):
    for key in (f"{country}|{crop}", f"crop|{crop}", "global"):
        if key in thresholds:
            return thresholds[key]["q20"], key
    raise AssertionError


def stratified_bootstrap_delta(rows, reps=3000):
    rng = np.random.default_rng(220199)
    groups = {}
    for i, row in enumerate(rows): groups.setdefault(row["group"], []).append(i)
    values = []
    for _ in range(reps):
        chosen = []
        for ids in groups.values(): chosen.extend(rng.choice(ids, len(ids), replace=True))
        y = np.asarray([rows[i]["target"] for i in chosen]); p = np.asarray([rows[i]["prediction"] for i in chosen]); b = np.asarray([rows[i]["baseline"] for i in chosen])
        values.append(np.sqrt(np.mean((y-p)**2)) - np.sqrt(np.mean((y-b)**2)))
    return [float(x) for x in np.quantile(values, [.025, .5, .975])]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    canonical = OUT / "report.json"
    if canonical.exists() and not (OUT / "report_pre_dual_tree.json").exists():
        (OUT / "report_pre_dual_tree.json").write_text(canonical.read_text(encoding="utf-8"), encoding="utf-8")
    data = np.load(CACHE, allow_pickle=False)
    target = data["target"].astype(float)
    reference = np.flatnonzero((data["role"] == "adapter_reference") & np.isfinite(target))
    challenge = np.flatnonzero((data["role"] == "locked_external") & np.isfinite(target))
    roles = np.asarray([subrole(x) if data["role"][i] == "adapter_reference" else "development_challenge" for i, x in enumerate(data["farm_group"])])
    fit = reference[roles[reference] == "model_fit"]
    early = reference[roles[reference] == "early_stop"]
    calibration = reference[roles[reference] == "conformal"]
    role_farms = {name: set(map(str, data["farm_group"][reference[roles[reference] == name]])) for name in ("model_fit", "early_stop", "conformal")}
    farm_disjoint = all(role_farms[a].isdisjoint(role_farms[b]) for a in role_farms for b in role_farms if a < b)
    countries = sorted(set(map(str, data["country"][fit]))); crops = sorted(set(map(str, data["crop"][fit])))

    milestones = {}
    for crop in crops:
        maxima = [float(np.nanmax(data["dynamic"][i, :data["lengths"][i], 17])) for i in fit if str(data["crop"][i]) == crop]
        median = float(np.median(maxima))
        milestones[crop] = {"training_fields": len(maxima), "median_training_max_gdd": median,
                            "looks_gdd": [float(x) for x in median * LOOK_FRACTIONS]}
    (OUT / "milestones.json").write_text(json.dumps(milestones, indent=2), encoding="utf-8")
    stats, thresholds = group_stats(data, fit)
    (OUT / "target_reference.json").write_text(json.dumps({"normalization": stats, "low_yield_thresholds": thresholds}, indent=2), encoding="utf-8")

    prediction_rows, look_reports = [], {}
    for look_index in range(3):
        look_map = {crop: milestones[crop]["looks_gdd"][look_index] for crop in crops}
        train_x, train_idx = encode_features(data, fit, look_map, countries, crops)
        stop_x, stop_idx = encode_features(data, early, look_map, countries, crops)
        test_x, test_idx = encode_features(data, challenge, look_map, countries, crops)
        medians = np.nanmedian(train_x, axis=0); medians[~np.isfinite(medians)] = 0.0
        train_x = np.where(np.isfinite(train_x), train_x, medians)
        stop_x = np.where(np.isfinite(stop_x), stop_x, medians)
        test_x = np.where(np.isfinite(test_x), test_x, medians)
        z = np.asarray([(target[i] - stats[f"{data['country'][i]}|{data['crop'][i]}"]["mean"]) / stats[f"{data['country'][i]}|{data['crop'][i]}"]["std"] for i in train_idx])
        model = ExtraTreesRegressor(n_estimators=600, min_samples_leaf=3, max_features=.7, bootstrap=False, random_state=2201, n_jobs=-1)
        model.fit(train_x, z)
        hist_model = HistGradientBoostingRegressor(max_iter=300, min_samples_leaf=15, l2_regularization=1.0, random_state=5501)
        hist_model.fit(train_x, z)
        stop_z = np.asarray([(target[i] - stats[f"{data['country'][i]}|{data['crop'][i]}"]["mean"]) / stats[f"{data['country'][i]}|{data['crop'][i]}"]["std"] for i in stop_idx])
        stop_et, stop_hist = model.predict(stop_x), hist_model.predict(stop_x)
        weights = np.linspace(0, 1, 11)
        hist_weight = float(min(weights, key=lambda w: float(np.sqrt(np.mean((stop_z - ((1-w)*stop_et + w*stop_hist))**2)))))
        pred_z = (1-hist_weight)*model.predict(test_x) + hist_weight*hist_model.predict(test_x)
        rows = []
        for i, value in zip(test_idx, pred_z):
            key = f"{data['country'][i]}|{data['crop'][i]}"; stat = stats[key]
            pred = float(value * stat["std"] + stat["mean"]); baseline = stat["mean"]
            cutoff, cutoff_key = threshold_for(thresholds, str(data["country"][i]), str(data["crop"][i]))
            row = {"look": look_index + 1, "field_id": str(data["field_id"][i]), "country": str(data["country"][i]),
                   "crop": str(data["crop"][i]), "group": key, "gdd": float(look_map[str(data["crop"][i])]),
                   "target": float(target[i]), "prediction": pred, "baseline": baseline,
                   "low_yield_cutoff": cutoff, "cutoff_source": cutoff_key, "is_low_yield": int(target[i] <= cutoff),
                   "risk_score": float(cutoff - pred)}
            rows.append(row); prediction_rows.append(row)
        y = np.asarray([r["target"] for r in rows]); p = np.asarray([r["prediction"] for r in rows]); b = np.asarray([r["baseline"] for r in rows]); labels = np.asarray([r["is_low_yield"] for r in rows]); risk = np.asarray([r["risk_score"] for r in rows])
        per_group = {}
        for group in sorted(set(r["group"] for r in rows)):
            subset = [r for r in rows if r["group"] == group]
            if len(subset) >= 5:
                per_group[group] = metric(np.asarray([r["target"] for r in subset]), np.asarray([r["prediction"] for r in subset]))
        look_reports[str(look_index + 1)] = {
            "look_gdd_by_crop": look_map, "eligible_train": len(train_idx), "eligible_challenge": len(test_idx),
            "challenge_coverage": float(len(test_idx) / len(challenge)), "prediction": metric(y, p), "baseline": metric(y, b),
            "rmse_delta_bootstrap95": stratified_bootstrap_delta(rows),
            "low_yield_prevalence": float(labels.mean()), "auroc": float(roc_auc_score(labels, risk)),
            "auprc": float(average_precision_score(labels, risk)), "per_group": per_group,
            "macro_group_rmse": float(np.mean([x["rmse"] for x in per_group.values()])),
            "worst_group_r2_n_ge_10": float(min(x["r2"] for x in per_group.values() if x["n"] >= 10)),
        }
        joblib.dump({"model": model, "hist_model": hist_model, "hist_weight": hist_weight, "feature_medians": medians, "countries": countries, "crops": crops,
                     "look_gdd": look_map, "target_stats": stats}, OUT / f"model_look{look_index+1}.joblib")

    with (OUT / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0])); writer.writeheader(); writer.writerows(prediction_rows)
    late_ci = look_reports["3"]["rmse_delta_bootstrap95"]
    auroc_count = sum(look_reports[str(i)]["auroc"] >= .65 for i in range(1, 4))
    integrity = {"audit_targets_read": False, "future_observations_used": False, "farm_disjoint_internal_roles": farm_disjoint,
                 "role_counts": {"model_fit": len(fit), "early_stop": len(early), "conformal": len(calibration), "challenge": len(challenge)},
                 "challenge_farm_overlap_with_fit": len(set(map(str, data["farm_group"][challenge])) & role_farms["model_fit"]),
                 "legacy_overlap_disclosed": True}
    gates = {
        "integrity": bool(farm_disjoint), "coverage_each_look_ge_0.80": all(look_reports[str(i)]["challenge_coverage"] >= .8 for i in range(1,4)),
        "late_rmse_bootstrap_improves_climatology": bool(late_ci[2] < 0),
        "late_rmse_not_worse_than_early": bool(look_reports["3"]["prediction"]["rmse"] <= look_reports["1"]["prediction"]["rmse"]),
        "auroc_ge_0.65_at_two_looks": bool(auroc_count >= 2),
    }
    report = {"experiment": 2, "attempt": 2, "predictor": "early_stop_selected_dual_tree", "status": "PASS" if all(gates.values()) else "FAIL", "look_reports": look_reports,
              "integrity": integrity, "pass_gates": gates}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True))


if __name__ == "__main__":
    main()
