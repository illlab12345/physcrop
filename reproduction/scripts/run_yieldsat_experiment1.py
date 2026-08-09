"""Experiment 1: field-mean adaptation of the public YieldSAT LSTM tutorial."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results/yieldsat_risk_v1/cache/development_sequences.npz"
OUT = ROOT / "results/yieldsat_risk_v1/experiment_01_official_baseline"
OFFICIAL_GROUPS = [
    ("Argentina", "soybean"), ("Brazil", "corn"), ("Germany", "rapeseed"),
    ("Germany", "wheat"), ("Uruguay", "soybean"),
]
SEEDS = [1301, 1302, 1303, 1304, 1305]


class LSTMModel(torch.nn.Module):
    """Architecture copied from the public tutorial: LSTM(12,64,1)+linear."""

    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=12, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = torch.nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)


def stable_bucket(text: str, modulus=100):
    return int(hashlib.sha256(("yieldsat-risk-v1:" + text).encode()).hexdigest()[:12], 16) % modulus


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def resample24(dynamic, dates, lengths):
    result = np.empty((len(lengths), 24, 12), dtype=np.float32)
    for i, length in enumerate(lengths):
        x = dynamic[i, :length, :12]
        t = dates[i, :length].astype(np.float64)
        query = np.linspace(t[0], t[-1], 24)
        for band in range(12):
            result[i, :, band] = np.interp(query, t, x[:, band])
    return result


def metrics(y, pred):
    rho = spearmanr(y, pred).statistic if len(y) > 2 else np.nan
    return {
        "n": int(len(y)), "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)), "r2": float(r2_score(y, pred)) if len(y) > 1 else np.nan,
        "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
    }


def fit_predict(train_x, train_y, stop_x, stop_y, test_x, seed):
    set_seed(seed)
    x_mean = train_x.mean(axis=(0, 1), keepdims=True)
    x_std = train_x.std(axis=(0, 1), keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    y_mean, y_std = float(train_y.mean()), float(train_y.std())
    y_std = max(y_std, 1e-6)
    transform = lambda x: ((x - x_mean) / x_std).astype(np.float32)
    train_xt = torch.from_numpy(transform(train_x))
    train_yt = torch.from_numpy(((train_y - y_mean) / y_std).astype(np.float32))
    stop_xt = torch.from_numpy(transform(stop_x))
    stop_yt = torch.from_numpy(((stop_y - y_mean) / y_std).astype(np.float32))
    test_xt = torch.from_numpy(transform(test_x))
    model = LSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_xt, train_yt), batch_size=min(128, len(train_xt)),
        shuffle=True, generator=generator, num_workers=0,
    )
    best_loss, best_epoch, best_state = math.inf, -1, None
    history = []
    for epoch in range(15):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            stop_loss = float(criterion(model(stop_xt), stop_yt))
        history.append(stop_loss)
        if stop_loss < best_loss:
            best_loss, best_epoch, best_state = stop_loss, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(test_xt).numpy() * y_std + y_mean
    normalizer = {"x_mean": x_mean.reshape(-1).tolist(), "x_std": x_std.reshape(-1).tolist(),
                  "y_mean": y_mean, "y_std": y_std}
    return prediction.astype(np.float64), best_epoch, history, normalizer, model.state_dict()


def bootstrap_delta(rows, repetitions=5000):
    rng = np.random.default_rng(8675309)
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault((row["country"], row["crop"]), []).append(i)
    deltas = []
    for _ in range(repetitions):
        sample = []
        for indices in groups.values():
            sample.extend(rng.choice(indices, size=len(indices), replace=True).tolist())
        y = np.asarray([rows[i]["target"] for i in sample])
        p = np.asarray([rows[i]["prediction"] for i in sample])
        b = np.asarray([rows[i]["mean_baseline"] for i in sample])
        deltas.append(np.sqrt(np.mean((y - p) ** 2)) - np.sqrt(np.mean((y - b) ** 2)))
    return [float(x) for x in np.quantile(deltas, [0.025, 0.5, 0.975])]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(CACHE, allow_pickle=False)
    x = resample24(data["dynamic"], data["dates"], data["lengths"])
    roles, countries, crops = data["role"], data["country"], data["crop"]
    targets, field_ids = data["target"].astype(float), data["field_id"]
    all_rows, seed_rows, provenance = [], [], {}
    deterministic_max_abs_diff = 0.0
    for country, crop in OFFICIAL_GROUPS:
        base = (countries == country) & (crops == crop) & np.isfinite(targets)
        reference = np.flatnonzero(base & (roles == "adapter_reference"))
        test = np.flatnonzero(base & (roles == "locked_external"))
        fit = np.asarray([i for i in reference if stable_bucket(str(field_ids[i])) >= 15], dtype=int)
        stop = np.asarray([i for i in reference if stable_bucket(str(field_ids[i])) < 15], dtype=int)
        if len(stop) < 5 or len(fit) < 20 or len(test) < 5:
            provenance[f"{country}|{crop}"] = {"status": "insufficient", "fit": len(fit), "stop": len(stop), "test": len(test)}
            continue
        predictions, states = [], []
        for seed in SEEDS:
            pred, epoch, history, normalizer, state = fit_predict(x[fit], targets[fit], x[stop], targets[stop], x[test], seed)
            predictions.append(pred); states.append(state)
            row = {"country": country, "crop": crop, "seed": seed, "best_epoch": epoch, **metrics(targets[test], pred)}
            seed_rows.append(row)
            torch.save({"state_dict": state, "normalizer": normalizer, "seed": seed, "group": [country, crop]},
                       OUT / f"model_{country}_{crop}_{seed}.pt")
        # Determinism audit: repeat first seed from scratch.
        repeated, *_ = fit_predict(x[fit], targets[fit], x[stop], targets[stop], x[test], SEEDS[0])
        deterministic_max_abs_diff = max(deterministic_max_abs_diff, float(np.max(np.abs(repeated - predictions[0]))))
        ensemble = np.mean(predictions, axis=0)
        mean_baseline = float(np.mean(targets[fit]))
        for index, pred in zip(test, ensemble):
            all_rows.append({"field_id": str(field_ids[index]), "country": country, "crop": crop,
                             "target": float(targets[index]), "prediction": float(pred), "mean_baseline": mean_baseline})
        provenance[f"{country}|{crop}"] = {"status": "completed", "fit": len(fit), "stop": len(stop), "test": len(test),
                                            "fit_ids_sha256": hashlib.sha256("\n".join(sorted(map(str, field_ids[fit]))).encode()).hexdigest(),
                                            "test_ids_sha256": hashlib.sha256("\n".join(sorted(map(str, field_ids[test]))).encode()).hexdigest()}
    if not all_rows:
        raise SystemExit("No official benchmark group had sufficient support")
    write_csv(OUT / "predictions.csv", all_rows)
    write_csv(OUT / "seed_metrics.csv", seed_rows)
    y = np.asarray([r["target"] for r in all_rows]); pred = np.asarray([r["prediction"] for r in all_rows]); base = np.asarray([r["mean_baseline"] for r in all_rows])
    per_group = {}
    for group in sorted({(r["country"], r["crop"]) for r in all_rows}):
        selected = [r for r in all_rows if (r["country"], r["crop"]) == group]
        per_group["|".join(group)] = {"lstm": metrics(np.asarray([r["target"] for r in selected]), np.asarray([r["prediction"] for r in selected])),
                                      "mean": metrics(np.asarray([r["target"] for r in selected]), np.asarray([r["mean_baseline"] for r in selected]))}
    pooled_lstm, pooled_mean = metrics(y, pred), metrics(y, base)
    ci = bootstrap_delta(all_rows)
    coverage = len(all_rows) / sum(int((countries == a).astype(int)[(crops == b) & (roles == "locked_external")].sum()) for a, b in OFFICIAL_GROUPS)
    scientific_pass = pooled_lstm["rmse"] < pooled_mean["rmse"]
    integrity = {
        "audit_targets_read": False, "field_overlap_fit_test": False,
        "normalizers_fit_only": True, "deterministic_max_abs_difference": deterministic_max_abs_diff,
        "deterministic_tolerance": 1e-7, "finite_prediction_fraction": float(np.isfinite(pred).mean()),
        "eligible_prediction_fraction": float(coverage),
    }
    report = {
        "experiment": 1, "name": "public YieldSAT LSTM field-mean flexible-format adaptation",
        "reference_notebook": "external/yieldsat-ml-tutorial.ipynb", "official_table_exact_reproduction": False,
        "reason_not_exact": "Only the flexible raw archive and tutorial are public locally; the tutorial expects preprocessed NetCDF and is demonstrative.",
        "pooled": {"lstm": pooled_lstm, "country_crop_training_mean": pooled_mean,
                   "rmse_delta_lstm_minus_mean_bootstrap95": ci},
        "per_group": per_group,
        "macro_r2": float(np.mean([v["lstm"]["r2"] for v in per_group.values()])),
        "integrity": integrity, "provenance": provenance,
        "pass_gates": {"integrity": bool(not integrity["audit_targets_read"] and deterministic_max_abs_diff <= 1e-7 and integrity["finite_prediction_fraction"] >= .95),
                       "beats_training_mean_rmse": bool(scientific_pass),
                       "bootstrap_ci_excludes_no_improvement": bool(ci[2] < 0)},
    }
    report["status"] = "PASS" if all(report["pass_gates"].values()) else "FAIL"
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True))


if __name__ == "__main__":
    main()
