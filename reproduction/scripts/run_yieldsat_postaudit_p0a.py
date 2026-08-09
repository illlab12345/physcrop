"""Post-audit P0-A diagnostics; never modifies the frozen final audit."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import beta
from sklearn.metrics import average_precision_score, mean_squared_error, r2_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/yieldsat_risk_v1"
OUT = BASE / "post_audit_p0p1"


def cp95(successes: int, trials: int):
    if trials == 0:
        return None
    lo = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, trials - successes + 1))
    hi = 1.0 if successes == trials else float(beta.ppf(0.975, successes + 1, trials - successes))
    return [lo, hi]


def classification_summary(truth, alert):
    truth = np.asarray(truth, dtype=bool); alert = np.asarray(alert, dtype=bool)
    tp = int(np.sum(truth & alert)); fp = int(np.sum(~truth & alert))
    fn = int(np.sum(truth & ~alert)); tn = int(np.sum(~truth & ~alert))
    return {
        "n": int(len(truth)), "low_n": int(np.sum(truth)), "normal_n": int(np.sum(~truth)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "fpr": fp / max(1, fp + tn), "fpr_cp95": cp95(fp, fp + tn),
        "sensitivity": tp / max(1, tp + fn), "sensitivity_cp95": cp95(tp, tp + fn),
        "precision": tp / max(1, tp + fp), "precision_cp95": cp95(tp, tp + fp),
    }


def parse_date(value):
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(str(value), fmt).date()
        except (TypeError, ValueError):
            pass
    raise ValueError(f"Unparseable date: {value!r}")


def paired_look_bootstrap(by_id, common, later, reps=5000):
    groups = {}
    for fid in common:
        groups.setdefault(by_id[fid][1]["group"], []).append(fid)
    rng = np.random.default_rng(20260802); values = {"rmse": [], "auroc": [], "auprc": []}
    for _ in range(reps):
        sampled = []
        for ids in groups.values(): sampled.extend(rng.choice(ids, len(ids), replace=True))
        y = np.asarray([float(by_id[f][1]["target"]) for f in sampled])
        label = np.asarray([int(by_id[f][1]["true_low"]) for f in sampled])
        p1 = np.asarray([float(by_id[f][1]["prediction"]) for f in sampled])
        pk = np.asarray([float(by_id[f][later]["prediction"]) for f in sampled])
        cutoff = np.asarray([float(by_id[f][1]["cutoff"]) for f in sampled])
        values["rmse"].append(float(np.sqrt(mean_squared_error(y, pk))-np.sqrt(mean_squared_error(y, p1))))
        values["auroc"].append(float(roc_auc_score(label, cutoff-pk)-roc_auc_score(label, cutoff-p1)))
        values["auprc"].append(float(average_precision_score(label, cutoff-pk)-average_precision_score(label, cutoff-p1)))
    return {f"{name}_later_minus_look1_cp95": [float(x) for x in np.quantile(v,[.025,.5,.975])] for name,v in values.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (BASE / "final_audit/final_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {}
    for row in rows:
        by_id.setdefault(row["field_id"], {})[int(row["look"])] = row

    field_ids = sorted(by_id)
    truth = np.asarray([int(by_id[f][min(by_id[f])]["true_low"]) for f in field_ids])
    watch = np.asarray([any(int(r["watch"]) for r in by_id[f].values()) for f in field_ids])
    action = np.asarray([int(next(iter(by_id[f].values()))["action_alert"]) for f in field_ids])
    policies = {"watch_any_hit_original_frozen": classification_summary(truth, watch),
                "action_two_consecutive_original_frozen": classification_summary(truth, action)}

    common = sorted(f for f, looks in by_id.items() if set(looks) == {1, 2, 3})
    common_reports = {}
    for look in (1, 2, 3):
        subset = [by_id[f][look] for f in common]
        y = np.asarray([float(r["target"]) for r in subset])
        p = np.asarray([float(r["prediction"]) for r in subset])
        labels = np.asarray([int(r["true_low"]) for r in subset])
        risk = np.asarray([float(r["cutoff"]) - float(r["prediction"]) for r in subset])
        common_reports[str(look)] = {
            "n": len(subset), "prevalence": float(np.mean(labels)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))), "r2": float(r2_score(y, p)),
            "auroc": float(roc_auc_score(labels, risk)),
            "auprc": float(average_precision_score(labels, risk)),
        }

    archive = np.load(BASE / "cache/audit_features_blind.npz", allow_pickle=False)
    audit = {k: archive[k] for k in archive.files}; archive.close()
    cache_ids = list(map(str, audit["field_id"])); cache_pos = {x: i for i, x in enumerate(cache_ids)}
    if set(field_ids) != set(cache_ids):
        raise RuntimeError("Audit field IDs differ from immutable blind cache")
    milestones = json.loads((BASE / "experiment_02_early_curve/milestones.json").read_text(encoding="utf-8"))
    lead_reports = {}
    lead_rows = []
    for look in (1, 2, 3):
        leads = []; crops = {}
        for fid in field_ids:
            if look not in by_id[fid]:
                continue
            i = cache_pos[fid]; crop = str(audit["crop"][i]); country = str(audit["country"][i])
            gdd_limit = float(milestones[crop]["looks_gdd"][look - 1])
            n = int(audit["lengths"][i]); gdd = audit["dynamic"][i, :n, 17].astype(float)
            use = np.flatnonzero(np.isfinite(gdd) & (gdd <= gdd_limit + 1e-6))
            if len(use) < 3:
                raise RuntimeError(f"Eligible prediction lacks three observations: {fid}, look {look}")
            last_day = datetime.fromordinal(int(audit["dates"][i, use[-1]])).date()
            meta_path = next((ROOT / "YieldSAT_raw_data" / country / fid).glob("metadata-*.json"))
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            harvest = parse_date(meta["harvesting_date"])
            lead = int((harvest - last_day).days)
            leads.append(lead); crops.setdefault(crop, []).append(lead)
            lead_rows.append({"field_id": fid, "look": look, "crop": crop, "last_acquisition": last_day.isoformat(),
                              "harvest": harvest.isoformat(), "lead_days": lead})
        q = np.quantile(leads, [.05, .25, .5, .75, .95])
        lead_reports[str(look)] = {
            "n": len(leads), "q05_q25_median_q75_q95_days": [float(x) for x in q],
            "minimum_days": int(min(leads)), "nonpositive_n": int(np.sum(np.asarray(leads) <= 0)),
            "crop_median_days": {c: float(np.median(v)) for c, v in sorted(crops.items())},
        }

    dev = np.load(BASE / "cache/development_sequences.npz", allow_pickle=False)
    ref_farms = set(map(str, dev["farm_group"][dev["role"] == "adapter_reference"])); dev.close()
    audit_farms = [str(audit["farm_group"][cache_pos[f]]) for f in field_ids]
    seen_mask = np.asarray([x in ref_farms for x in audit_farms])
    farm_overlap = {
        "audit_fields": len(field_ids), "seen_farm_fields": int(seen_mask.sum()),
        "unseen_farm_fields": int((~seen_mask).sum()), "audit_farms": len(set(audit_farms)),
        "seen_farms": len(set(x for x in audit_farms if x in ref_farms)),
        "unseen_farms": len(set(x for x in audit_farms if x not in ref_farms)),
        "adapter_reference_farms": len(ref_farms),
        "inference": "insufficient unseen-audit support" if (~seen_mask).sum() < 20 else "stratified estimates permitted",
    }
    with (OUT / "p0a_lead_times.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(lead_rows[0])); writer.writeheader(); writer.writerows(lead_rows)
    paired_changes = {str(k): paired_look_bootstrap(by_id, common, k) for k in (2, 3)}
    report = {
        "analysis_status": "post-audit diagnostic; original frozen audit unchanged",
        "integrity": {"audit_id_match": True, "audit_outcomes_used_for_training_or_selection": False},
        "original_frozen_policy_intervals": policies,
        "common_three_look_cohort": {"field_ids_n": len(common), "looks": common_reports,
                                      "paired_change_from_look1": paired_changes},
        "preharvest_lead_time": lead_reports, "farm_overlap": farm_overlap,
    }
    (OUT / "p0a_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
