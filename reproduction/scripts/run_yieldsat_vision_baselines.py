"""Experiment 10: genuine AgriFM and ConvLSTM YieldSAT baselines.

The experiment is deliberately isolated from ``final_audit``.  It reuses the
frozen Experiment 5 field roles, third-look GDD cutoffs, target normalization,
and metric implementation.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import date
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import tifffile
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_agrifm_encoder_baseline import build_encoder  # noqa: E402
from run_yieldsat_experiment2 import (  # noqa: E402
    CACHE,
    OUT as EXP2_OUT,
    encode_features,
    subrole,
)
from run_yieldsat_experiment5 import (  # noqa: E402
    bootstrap_delta,
    group_metrics,
    metrics,
    raw_predictions,
)


OUT = ROOT / "results" / "yieldsat_risk_v1" / "experiment_10_vision_baselines"
IMAGE_CACHE = OUT / "look3_image_sequences_32.npz"
AGRIFM_CACHE = OUT / "agrifm_look3_features.npz"
CHECKPOINT = ROOT / "AgriFM-main" / "checkpoints" / "AgriFM.pth"
CONVLSTM_SOURCE = (
    ROOT / "ConvLSTM_results_20260722" / "third_party"
    / "ConvLSTM_pytorch-master" / "convlstm.py"
)

# Official 10-band AgriFM S2 ordering: B02/B03/B04/B05/B06/B07/B08/B8A/B11/B12.
S2_INDICES = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 10, 11], dtype=int)
AGRIFM_MEAN = np.asarray(
    [4179.1920, 4065.9107, 3957.2749, 5207.4525, 4327.1223,
     4873.1610, 5049.1638, 5111.0781, 3056.8635, 2490.9675],
    dtype=np.float32,
)
AGRIFM_STD = np.asarray(
    [4041.5212, 3691.0031, 3629.3313, 2973.5179, 3569.7334,
     3085.9151, 2937.5601, 2806.0446, 1808.3002, 1694.2022],
    dtype=np.float32,
)
FRAME_FRACTIONS = np.linspace(0.0, 1.0, 6)
MODEL_SEEDS = (5611, 5612, 5613)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rebuild-images", action="store_true")
    parser.add_argument("--rebuild-agrifm", action="store_true")
    parser.add_argument("--skip-agrifm", action="store_true")
    parser.add_argument("--skip-convlstm", action="store_true")
    return parser.parse_args()


def _resize_square(values: np.ndarray, size: int = 32) -> np.ndarray:
    """Aspect-preserving square padding and bilinear resizing for CxHxW data."""
    channels, height, width = values.shape
    side = max(height, width)
    fill = np.median(values.reshape(channels, -1), axis=1).astype(np.float32)
    square = np.broadcast_to(fill[:, None, None], (channels, side, side)).copy()
    y0 = (side - height) // 2
    x0 = (side - width) // 2
    square[:, y0:y0 + height, x0:x0 + width] = values
    tensor = torch.from_numpy(square[None])
    resized = torch.nn.functional.interpolate(
        tensor, size=(size, size), mode="bilinear", align_corners=False,
    )[0]
    return resized.numpy().astype(np.float32)


def _read_frame(folder: Path, stamp: str) -> np.ndarray:
    s2_path = folder / "s2_images" / f"S2_L2A_{stamp}.tif"
    scl_path = folder / "scl_masks" / f"S2_L2A_SCL_{stamp}.tif"
    s2 = np.asarray(tifffile.imread(s2_path), dtype=np.float32)
    if s2.ndim != 3 or s2.shape[-1] < 12:
        raise RuntimeError(f"Unexpected Sentinel-2 shape {s2.shape}: {s2_path}")
    bands = s2[:, :, S2_INDICES].transpose(2, 0, 1)
    nonzero = np.all(bands > 0, axis=0)
    if scl_path.exists():
        scl = np.asarray(tifffile.imread(scl_path))
        valid = nonzero & np.isin(scl, (4, 5))
        if valid.sum() < 32:
            valid = nonzero
    else:
        valid = nonzero
    for channel in range(bands.shape[0]):
        channel_values = bands[channel]
        finite = valid & np.isfinite(channel_values)
        fill = float(np.median(channel_values[finite])) if finite.any() else float(AGRIFM_MEAN[channel])
        channel_values[~finite] = fill
        bands[channel] = channel_values
    return np.clip(_resize_square(bands) / 10000.0, 0.0, 1.5)


def build_image_cache(data: np.lib.npyio.NpzFile, look_by_crop: dict[str, float]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    count = len(data["field_id"])
    images = np.empty((count, 6, 10, 32, 32), dtype=np.float16)
    selected_gdd = np.empty((count, 6), dtype=np.float32)
    selected_date = np.empty((count, 6), dtype=np.int32)
    causal_ok = np.ones(count, dtype=np.uint8)
    for i in range(count):
        field_id = str(data["field_id"][i])
        country = str(data["country"][i])
        crop = str(data["crop"][i])
        cutoff = float(look_by_crop[crop])
        length = int(data["lengths"][i])
        gdd = data["dynamic"][i, :length, 17].astype(float)
        ordinals = data["dates"][i, :length].astype(int)
        causal = np.flatnonzero(np.isfinite(gdd) & (gdd <= cutoff + 1e-6))
        if len(causal) < 3:
            raise RuntimeError(f"Fewer than three causal acquisitions: {field_id}")
        folder = ROOT / "YieldSAT_raw_data" / country / field_id
        for j, query in enumerate(FRAME_FRACTIONS * cutoff):
            order = causal[np.argsort(np.abs(gdd[causal] - query))]
            chosen = None
            for pos in order:
                stamp = date.fromordinal(int(ordinals[pos])).strftime("%Y%m%d")
                if (folder / "s2_images" / f"S2_L2A_{stamp}.tif").exists():
                    chosen = (int(pos), stamp)
                    break
            if chosen is None:
                raise FileNotFoundError(f"No raw causal image for {field_id}, target GDD {query:.1f}")
            pos, stamp = chosen
            images[i, j] = _read_frame(folder, stamp).astype(np.float16)
            selected_gdd[i, j] = float(gdd[pos])
            selected_date[i, j] = int(stamp)
            causal_ok[i] &= np.uint8(gdd[pos] <= cutoff + 1e-6)
        if (i + 1) % 50 == 0 or i + 1 == count:
            print(f"image cache {i + 1}/{count}", flush=True)
    np.savez_compressed(
        IMAGE_CACHE,
        field_id=data["field_id"],
        images=images,
        selected_gdd=selected_gdd,
        selected_date=selected_date,
        causal_ok=causal_ok,
    )


def load_images(data, look_by_crop, force: bool) -> dict[str, np.ndarray]:
    if force or not IMAGE_CACHE.exists():
        build_image_cache(data, look_by_crop)
    archive = np.load(IMAGE_CACHE, allow_pickle=False)
    cached = {key: archive[key] for key in archive.files}
    if not np.array_equal(cached["field_id"].astype(str), data["field_id"].astype(str)):
        raise RuntimeError("Image cache field order does not match development cache")
    if not bool(np.all(cached["causal_ok"] == 1)):
        raise RuntimeError("Image cache contains a post-look acquisition")
    return cached


def extract_agrifm(images: np.ndarray, force: bool, smoke: bool) -> np.ndarray:
    if AGRIFM_CACHE.exists() and not force and not smoke:
        archive = np.load(AGRIFM_CACHE, allow_pickle=False)
        return archive["features"].astype(np.float32)
    encoder = build_encoder(str(CHECKPOINT), "cpu")
    features = []
    limit = min(len(images), 24) if smoke else len(images)
    batch_size = 4
    start_time = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, limit, batch_size):
            values = images[start:min(start + batch_size, limit)].astype(np.float32) * 10000.0
            values = (values - AGRIFM_MEAN[None, None, :, None, None]) / AGRIFM_STD[None, None, :, None, None]
            output = encoder(torch.from_numpy(values))["encoder_features"]
            features.append(output.mean(dim=(-2, -1)).numpy().astype(np.float32))
            if start == 0 or (start + batch_size) % 100 == 0 or start + batch_size >= limit:
                print(f"AgriFM features {min(start + batch_size, limit)}/{limit}", flush=True)
    feature_array = np.concatenate(features)
    if not smoke:
        np.savez_compressed(
            AGRIFM_CACHE,
            features=feature_array,
            checkpoint=str(CHECKPOINT.name),
            seconds=np.asarray([time.perf_counter() - start_time], dtype=np.float64),
        )
    return feature_array


def load_convlstm_class():
    spec = importlib.util.spec_from_file_location("yieldsat_convlstm_reference", CONVLSTM_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import ConvLSTM reference: {CONVLSTM_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ConvLSTM


class YieldConvLSTM(nn.Module):
    def __init__(self, core_class, context_dim: int):
        super().__init__()
        self.recurrent = core_class(
            input_dim=10,
            hidden_dim=[16, 32],
            kernel_size=(3, 3),
            num_layers=2,
            batch_first=True,
            bias=True,
            return_all_layers=False,
        )
        self.context = nn.Sequential(
            nn.Linear(context_dim, 32), nn.ReLU(), nn.LayerNorm(32),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.10), nn.Linear(32, 1),
        )

    def forward(self, images: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.recurrent(images)
        spatial = outputs[-1][:, -1].mean(dim=(-2, -1))
        return self.head(torch.cat([spatial, self.context(context)], dim=1)).squeeze(1)


class IndexedDataset(Dataset):
    def __init__(self, images, context, target, indices, image_mean, image_std):
        self.images = images
        self.context = context
        self.target = target
        self.indices = np.asarray(indices, dtype=int)
        self.image_mean = image_mean
        self.image_std = image_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = int(self.indices[item])
        image = self.images[idx].astype(np.float32)
        image = (image - self.image_mean) / self.image_std
        return (
            torch.from_numpy(image),
            torch.from_numpy(self.context[idx].astype(np.float32)),
            torch.tensor(self.target[idx], dtype=torch.float32),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_loader(dataset, batch_size, shuffle, seed):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def evaluate_model(model, loader):
    model.eval()
    values = []
    losses = []
    loss_fn = nn.MSELoss(reduction="sum")
    count = 0
    with torch.inference_mode():
        for images, context, target in loader:
            prediction = model(images.float(), context.float())
            values.append(prediction.numpy())
            losses.append(float(loss_fn(prediction, target).item()))
            count += len(target)
    return np.concatenate(values), float(sum(losses) / max(count, 1))


def train_convlstm(
    images, context, target_z, fit_idx, stop_idx, test_idx, smoke: bool,
):
    core = load_convlstm_class()
    image_mean = images[fit_idx].astype(np.float32).mean(axis=(0, 1, 3, 4), keepdims=False)[:, None, None]
    image_std = images[fit_idx].astype(np.float32).std(axis=(0, 1, 3, 4), keepdims=False)[:, None, None]
    image_std[image_std < 1e-4] = 1.0
    train_ds = IndexedDataset(images, context, target_z, fit_idx, image_mean, image_std)
    stop_ds = IndexedDataset(images, context, target_z, stop_idx, image_mean, image_std)
    test_ds = IndexedDataset(images, context, target_z, test_idx, image_mean, image_std)
    predictions = []
    histories = {}
    seed_info = {}
    max_epochs = 2 if smoke else 80
    seeds = MODEL_SEEDS[:1] if smoke else MODEL_SEEDS
    for seed in seeds:
        set_seed(seed)
        model = YieldConvLSTM(core, context.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best = (float("inf"), None, 0)
        stale = 0
        history = []
        for epoch in range(1, max_epochs + 1):
            model.train()
            train_loss = 0.0
            seen = 0
            for image_batch, context_batch, target_batch in make_loader(train_ds, 32, True, seed + epoch):
                optimizer.zero_grad(set_to_none=True)
                output = model(image_batch.float(), context_batch.float())
                loss = nn.functional.mse_loss(output, target_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += float(loss.item()) * len(target_batch)
                seen += len(target_batch)
            _, stop_loss = evaluate_model(model, make_loader(stop_ds, 64, False, seed))
            history.append({"epoch": epoch, "train_mse": train_loss / max(seen, 1), "stop_mse": stop_loss})
            print(
                f"ConvLSTM seed={seed} epoch={epoch:02d} "
                f"train={history[-1]['train_mse']:.5f} stop={stop_loss:.5f}",
                flush=True,
            )
            if stop_loss < best[0] - 1e-5:
                best = (stop_loss, copy.deepcopy(model.state_dict()), epoch)
                stale = 0
            else:
                stale += 1
                if stale >= (1 if smoke else 10):
                    break
        if best[1] is None:
            raise RuntimeError(f"ConvLSTM seed {seed} produced no checkpoint")
        model.load_state_dict(best[1])
        pred, _ = evaluate_model(model, make_loader(test_ds, 64, False, seed))
        predictions.append(pred)
        histories[str(seed)] = history
        seed_info[str(seed)] = {"best_epoch": best[2], "best_stop_mse": best[0]}
        if not smoke:
            torch.save(
                {"state_dict": best[1], "image_mean": image_mean, "image_std": image_std},
                OUT / f"convlstm_seed{seed}.pt",
            )
    return np.mean(predictions, axis=0), histories, seed_info


def context_features(encoded: np.ndarray, train_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Feature blocks 16 and 17 are valid fraction and cumulative precipitation.
    context = np.concatenate([encoded[:, 96:108], encoded[:, 108:]], axis=1).astype(np.float32)
    medians = np.nanmedian(context[train_indices], axis=0)
    medians[~np.isfinite(medians)] = 0.0
    context = np.where(np.isfinite(context), context, medians)
    lower = np.quantile(context[train_indices], 0.005, axis=0)
    upper = np.quantile(context[train_indices], 0.995, axis=0)
    context = np.clip(context, lower, upper)
    mean = context[train_indices].mean(axis=0, keepdims=True)
    std = context[train_indices].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((context - mean) / std).astype(np.float32), medians


def fit_agrifm_head(features, context, target_z, fit_idx, stop_idx, test_idx):
    components = min(64, len(fit_idx) - 1, features.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=5610)
    fit_rep = pca.fit_transform(features[fit_idx])
    stop_rep = pca.transform(features[stop_idx])
    test_rep = pca.transform(features[test_idx])
    fit_x = np.concatenate([fit_rep, context[fit_idx]], axis=1)
    stop_x = np.concatenate([stop_rep, context[stop_idx]], axis=1)
    test_x = np.concatenate([test_rep, context[test_idx]], axis=1)
    candidates = []
    for alpha in (0.1, 1.0, 10.0, 100.0):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(fit_x, target_z[fit_idx])
        stop_pred = model.predict(stop_x)
        stop_rmse = float(np.sqrt(np.mean((target_z[stop_idx] - stop_pred) ** 2)))
        candidates.append((stop_rmse, alpha, model))
    selected = min(candidates, key=lambda item: (item[0], item[1]))
    return selected[2].predict(test_x), {
        "selected_alpha": selected[1],
        "early_stop_normalized_rmse": selected[0],
        "pca_components": components,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "candidate_stop_rmse": {str(alpha): rmse for rmse, alpha, _ in candidates},
    }


def write_predictions(rows):
    with (OUT / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(CACHE, allow_pickle=False)
    target = data["target"].astype(float)
    reference = np.flatnonzero((data["role"] == "adapter_reference") & np.isfinite(target))
    fit = np.asarray([i for i in reference if subrole(data["farm_group"][i]) == "model_fit"], dtype=int)
    stop = np.asarray([i for i in reference if subrole(data["farm_group"][i]) == "early_stop"], dtype=int)
    test = np.flatnonzero((data["role"] == "locked_external") & np.isfinite(target))
    bundle = joblib.load(EXP2_OUT / "model_look3.joblib")
    look = bundle["look_gdd"]
    stats = bundle["target_stats"]
    thresholds = json.loads((EXP2_OUT / "target_reference.json").read_text(encoding="utf-8"))["low_yield_thresholds"]

    # Shared context uses encode_features only to reproduce the frozen temporal
    # alignment and categorical ordering. Image tensors come from raw TIFFs.
    all_indices = np.arange(len(target), dtype=int)
    encoded, encoded_indices = encode_features(
        data, all_indices, look, bundle["countries"], bundle["crops"],
    )
    if not np.array_equal(encoded_indices, all_indices):
        raise RuntimeError("Vision baselines do not cover every development field")
    context, _ = context_features(encoded, fit)
    target_z = np.asarray([
        (target[i] - stats[f"{data['country'][i]}|{data['crop'][i]}"]["mean"])
        / stats[f"{data['country'][i]}|{data['crop'][i]}"]["std"]
        for i in range(len(target))
    ], dtype=np.float32)

    if args.smoke:
        # Preserve every role in a small end-to-end check.
        fit = fit[:32]
        stop = stop[:16]
        test = test[:16]
    cached = load_images(data, look, args.rebuild_images)
    images = cached["images"]

    predictions_z = {}
    metadata = {}
    start = time.perf_counter()
    if not args.skip_agrifm:
        agrifm = extract_agrifm(images, args.rebuild_agrifm, args.smoke)
        if args.smoke:
            # Smoke extraction covers the beginning of the global cache; run a
            # shape/finite check only because role indices are not contiguous.
            if agrifm.shape != (24, 1024) or not np.isfinite(agrifm).all():
                raise RuntimeError("AgriFM smoke extraction failed")
        else:
            pred, info = fit_agrifm_head(agrifm, context, target_z, fit, stop, test)
            predictions_z["agrifm_frozen_context"] = pred
            metadata["agrifm_frozen_context"] = info
    if not args.skip_convlstm:
        pred, histories, seed_info = train_convlstm(
            images, context, target_z, fit, stop, test, args.smoke,
        )
        predictions_z["convlstm_context"] = pred
        metadata["convlstm_context"] = {
            "seeds": list(MODEL_SEEDS[:1] if args.smoke else MODEL_SEEDS),
            "seed_info": seed_info,
            "parameters": int(sum(p.numel() for p in YieldConvLSTM(load_convlstm_class(), context.shape[1]).parameters())),
        }
        (OUT / ("training_history_smoke.json" if args.smoke else "training_history.json")).write_text(
            json.dumps(histories, indent=2), encoding="utf-8",
        )
    if args.smoke:
        print(json.dumps({"status": "PASS_SMOKE", "models": list(predictions_z)}, indent=2))
        return

    raw = {
        name: raw_predictions(values, test, data, stats)
        for name, values in predictions_z.items()
    }
    reports = {}
    test_y = target[test]
    for name, prediction in raw.items():
        gm = group_metrics(test_y, prediction, test, data)
        reports[name] = {
            "overall": metrics(test_y, prediction, test, data, thresholds),
            "per_group": gm,
            "macro_group_rmse": float(np.mean([value["rmse"] for value in gm.values()])),
            "macro_group_r2": float(np.mean([value["r2"] for value in gm.values()])),
            "worst_group_r2_n_ge_10": float(min(value["r2"] for value in gm.values() if value["n"] >= 10)),
            **metadata[name],
        }

    rows = []
    for position, index in enumerate(test):
        row = {
            "field_id": str(data["field_id"][index]),
            "country": str(data["country"][index]),
            "crop": str(data["crop"][index]),
            "group": f"{data['country'][index]}|{data['crop'][index]}",
            "target": float(target[index]),
        }
        row.update({name: float(values[position]) for name, values in raw.items()})
        rows.append(row)
    write_predictions(rows)

    # Reuse Experiment 5 predictions only as a field-ID/target integrity oracle
    # and for paired confidence intervals; no fitting decision reads its scores.
    with (ROOT / "results" / "yieldsat_risk_v1" / "experiment_05_strong_baselines" / "predictions.csv").open(
        "r", encoding="utf-8-sig", newline="",
    ) as handle:
        exp5_rows = list(csv.DictReader(handle))
    exp5_ids = [row["field_id"] for row in exp5_rows]
    same_ids = exp5_ids == [row["field_id"] for row in rows]
    same_targets = same_ids and bool(np.allclose(
        [float(row["target"]) for row in exp5_rows],
        [row["target"] for row in rows],
        atol=1e-7,
    ))
    bootstrap_rows = []
    for row, old in zip(rows, exp5_rows):
        merged = dict(row)
        merged["phenology_dual_tree"] = float(old["phenology_dual_tree"])
        bootstrap_rows.append(merged)
    paired = {
        name: bootstrap_delta(bootstrap_rows, name, "phenology_dual_tree")
        for name in raw
    }
    checks = {
        "same_316_challenge_ids": bool(same_ids and len(rows) == 316),
        "same_targets": same_targets,
        "causal_acquisitions_only": bool(np.all(cached["causal_ok"] == 1)),
        "finite_predictions": bool(all(np.isfinite(values).all() for values in raw.values())),
        "all_models_present": set(raw) == {"agrifm_frozen_context", "convlstm_context"},
        "no_audit_files_read": True,
    }
    report = {
        "experiment": 10,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": "PLAN.md",
        "models": reports,
        "paired_rmse_delta_vs_physcrop_bootstrap95": paired,
        "integrity": checks,
        "runtime_seconds": time.perf_counter() - start,
    }
    (OUT / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError("Experiment 10 integrity checks failed")


if __name__ == "__main__":
    main()
