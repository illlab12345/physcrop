"""Generate publication-ready Figures 3 and 4 for the YieldSAT WACV paper."""
from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "yieldsat_risk_v1"
OUT = ROOT / "论文"

# Color-blind-safe, restrained scientific palette.
NAVY = "#244F73"
TEAL = "#1B9E77"
ORANGE = "#D98900"
VERMILLION = "#C84C36"
PURPLE = "#7A5195"
GRAY = "#737B83"
LIGHT_GRAY = "#D9DEE3"
INK = "#20262E"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.6,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.65,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def clean_axis(ax: plt.Axes, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=LIGHT_GRAY, linewidth=0.55, alpha=0.7, zorder=0)
    ax.tick_params(length=2.5, width=0.6)


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="top",
    )


def save_all(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 450}),
    ):
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.025, **kwargs)


def figure3() -> None:
    final = json.loads((RESULTS / "final_audit" / "final_report.json").read_text(encoding="utf-8"))
    clean = json.loads((RESULTS / "post_audit_p0p1" / "p0b_report.json").read_text(encoding="utf-8"))
    e4 = json.loads(
        (RESULTS / "experiment_04_familywise_alert" / "report.json").read_text(encoding="utf-8")
    )
    looks = ["1", "2", "3"]
    x = np.arange(1, 4)
    rmse = np.array([final["looks"][k]["metrics"]["rmse"] for k in looks])
    baseline = np.array([final["looks"][k]["baseline_rmse"] for k in looks])
    auroc = np.array([final["looks"][k]["metrics"]["auroc"] for k in looks])
    auprc = np.array([final["looks"][k]["metrics"]["auprc"] for k in looks])
    prevalence = np.array([final["looks"][k]["metrics"]["prevalence"] for k in looks])
    cov90 = np.array([final["looks"][k]["coverage_90"] for k in looks])
    cov95 = np.array([final["looks"][k]["coverage_95"] for k in looks])

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.72), constrained_layout=True)
    ax = axes[0, 0]
    clean_axis(ax)
    ax.plot(x, baseline, "--s", color=GRAY, linewidth=1.45, markersize=4.2, label="Country–crop climatology")
    ax.plot(x, rmse, "-o", color=NAVY, linewidth=1.9, markersize=4.7, label="PhysCrop-Risk")
    reductions = 100 * (baseline - rmse) / baseline
    for xi, y0, y1, pct in zip(x, rmse, baseline, reductions):
        ax.annotate("", xy=(xi, y0 + 0.015), xytext=(xi, y1 - 0.015), arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.9))
        ax.text(xi + 0.06, (y0 + y1) / 2, f"{pct:.0f}%", color=TEAL, fontsize=7, va="center")
    ax.set_xticks(x, ["Look 1", "Look 2", "Look 3"])
    ax.set_ylabel("RMSE (t ha$^{-1}$)")
    ax.set_ylim(0.96, 1.56)
    ax.set_xlim(0.85, 3.62)
    ax.text(3.10, baseline[-1], "Climatology", color=GRAY, va="center", fontsize=6.0)
    ax.text(3.10, rmse[-1], "PhysCrop-Risk", color=NAVY, va="center", fontsize=6.0)
    ax.set_title("Prediction improves with causal evidence", loc="left", pad=4, fontsize=7.2)
    panel_label(ax, "a")

    ax = axes[0, 1]
    clean_axis(ax)
    ax.plot(x, auroc, "-o", color=ORANGE, linewidth=1.9, markersize=4.7, label="AUROC")
    ax.plot(x, auprc, "-s", color=TEAL, linewidth=1.9, markersize=4.5, label="AUPRC")
    ax.plot(x, prevalence, ":", color=GRAY, linewidth=1.35)
    ax.text(3.05, auroc[-1], f"AUROC  {auroc[-1]:.3f}", color=ORANGE, va="center", fontsize=6.0)
    ax.text(3.05, auprc[-1], f"AUPRC  {auprc[-1]:.3f}", color=TEAL, va="center", fontsize=6.0)
    ax.set_xticks(x, ["Look 1", "Look 2", "Look 3"])
    ax.set_ylabel("Score")
    ax.set_ylim(0.16, 0.91)
    ax.text(
        3.08,
        prevalence[-1],
        f"Chance level = {prevalence[-1]:.3f}",
        color=GRAY,
        va="center",
        ha="left",
        fontsize=6.0,
    )
    ax.set_xlim(0.85, 4.02)
    ax.set_title("Low-yield discrimination strengthens", loc="left", pad=4, fontsize=7.2)
    panel_label(ax, "b")

    ax = axes[1, 0]
    clean_axis(ax)
    ax.axhline(0.90, color=TEAL, linestyle="--", linewidth=1.0, alpha=0.75)
    ax.axhline(0.95, color=NAVY, linestyle="--", linewidth=1.0, alpha=0.75)
    ax.plot(x, cov90, "-o", color=TEAL, linewidth=1.9, markersize=4.7, label="90% bound")
    ax.plot(x, cov95, "-s", color=NAVY, linewidth=1.9, markersize=4.5, label="95% bound")
    for xi, value in zip(x, cov90):
        ax.text(xi, value - 0.0105, f"{100*value:.1f}", ha="center", va="top", color=TEAL, fontsize=6.8)
    for xi, value in zip(x, cov95):
        ax.text(xi, value + 0.006, f"{100*value:.1f}", ha="center", va="bottom", color=NAVY, fontsize=6.8)
    ax.set_xticks(x, ["Look 1", "Look 2", "Look 3"])
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0.88, 1.008)
    ax.set_yticks([0.90, 0.925, 0.95, 0.975, 1.00], ["90", "92.5", "95", "97.5", "100"])
    ax.set_xlim(0.85, 3.58)
    ax.text(3.07, cov90[-1], "90% bound", color=TEAL, va="center", fontsize=6.0)
    ax.text(3.07, cov95[-1], "95% bound", color=NAVY, va="center", fontsize=6.0)
    ax.set_title("One-sided bounds remain conservative", loc="left", pad=4, fontsize=7.2)
    panel_label(ax, "c")

    ax = axes[1, 1]
    clean_axis(ax, grid="both")
    policies = [
        ("Marginal 0.05", e4["policy_summaries"]["marginal_05"], GRAY, "X"),
        ("Naïve repeated", e4["policy_summaries"]["naive"], ORANGE, "^"),
        ("Watch", e4["policy_summaries"]["alpha_spending"], NAVY, "s"),
        ("Persistent action", e4["policy_summaries"]["alpha_spending_persistent"], TEAL, "o"),
    ]
    for label, values, color, marker in policies:
        ax.scatter(
            100 * values["fpr_field_season"],
            100 * values["sensitivity"],
            s=43,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
            label=label,
        )
    audit = final["action_alert"]
    ax.scatter(
        100 * audit["fpr"],
        100 * audit["sensitivity"],
        s=78,
        marker="*",
        color=VERMILLION,
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
        label="Original frozen action",
    )
    clean_dual = clean["audit_uniform_policy_comparison"]["dual_tree"]
    ax.scatter(
        100 * clean_dual["watch"]["fpr"], 100 * clean_dual["watch"]["sensitivity"],
        s=63, marker="D", facecolor="white", edgecolor=VERMILLION, linewidth=1.15,
        zorder=5, label="Clean-null audit watch",
    )
    ax.scatter(
        100 * clean_dual["action"]["fpr"], 100 * clean_dual["action"]["sensitivity"],
        s=63, marker="D", color=VERMILLION, edgecolor="white", linewidth=0.7,
        zorder=6, label="Clean-null audit action",
    )
    ax.axvline(5, color=VERMILLION, linestyle="--", linewidth=1.0, alpha=0.75)
    ax.text(5.18, 61.5, "5% FPR target", color=VERMILLION, fontsize=6.0, va="top", ha="left")
    ax.set_xlabel("Field-season false-positive rate (%)")
    ax.set_ylabel("Sensitivity (%)")
    ax.set_xlim(-0.5, 13.0)
    ax.set_ylim(0, 65)
    ax.set_title("Persistence controls false alarms", loc="left", pad=4, fontsize=7.2)
    ax.legend(frameon=False, loc="lower right", ncol=1, handletextpad=0.35,
              labelspacing=0.18, fontsize=5.2)
    panel_label(ax, "d")

    save_all(fig, "figure3_early_prediction_and_alerting")
    plt.close(fig)


def read_metrics() -> list[dict[str, float | str]]:
    path = RESULTS / "experiment_07_spatial_maps" / "field_metrics.csv"
    rows: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, float | str] = {"field_id": raw["field_id"]}
            for key, value in raw.items():
                if key not in ("field_id", "country", "crop"):
                    row[key] = float(value)
            row["country"] = raw["country"]
            row["crop"] = raw["crop"]
            rows.append(row)
    return rows


def causal_rgb(field_id: str, target_mask: np.ndarray) -> np.ndarray:
    cache = np.load(RESULTS / "cache" / "development_sequences.npz", allow_pickle=False)
    field_ids = cache["field_id"].astype(str)
    matches = np.flatnonzero(field_ids == field_id)
    if len(matches) != 1:
        cache.close()
        raise RuntimeError(f"Cannot uniquely locate {field_id} in the development cache")
    idx = int(matches[0])
    country = str(cache["country"][idx])
    crop = str(cache["crop"][idx])
    length = int(cache["lengths"][idx])
    gdd = cache["dynamic"][idx, :length, 17]
    ordinals = cache["dates"][idx, :length]
    bundle = joblib.load(RESULTS / "experiment_02_early_curve" / "model_look3.joblib")
    cutoff = float(bundle["look_gdd"][crop])
    positions = np.flatnonzero(np.isfinite(gdd) & (gdd <= cutoff + 1e-6))
    folder = ROOT / "YieldSAT_raw_data" / country / field_id
    chosen = None
    for pos in positions[::-1]:
        stamp = date.fromordinal(int(ordinals[pos])).strftime("%Y%m%d")
        s2_path = folder / "s2_images" / f"S2_L2A_{stamp}.tif"
        scl_path = folder / "scl_masks" / f"S2_L2A_SCL_{stamp}.tif"
        if not s2_path.exists() or not scl_path.exists():
            continue
        scl = np.asarray(tifffile.imread(scl_path))
        valid = np.isin(scl, (4, 5)) & target_mask
        if valid.sum() < 50:
            continue
        chosen = (np.asarray(tifffile.imread(s2_path), dtype=np.float32) / 10000.0, valid)
        break
    cache.close()
    if chosen is None:
        raise RuntimeError(f"No causal clear Sentinel-2 image found for {field_id}")
    s2, valid = chosen
    rgb = s2[:, :, [3, 2, 1]]
    values = rgb[valid]
    lo, hi = np.nanpercentile(values, [2, 98])
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.82
    rgb[~target_mask] = 1.0
    return rgb


def short_field_label(field_id: str, country: str, crop: str, rho: float) -> str:
    field_token = next(part for part in field_id.split("_") if part.startswith("field"))
    return f"{country} · {crop}\n{field_token}   ρ={rho:.2f}"


def figure4() -> None:
    spatial_dir = RESULTS / "experiment_07_spatial_maps"
    metrics = read_metrics()
    by_id = {str(row["field_id"]): row for row in metrics}
    # All examples come from the outcome-blind, hash-selected qualitative set.
    # This subset is chosen for geographic/crop diversity rather than error rank.
    examples = [
        "Argentina_DUP1_farm2_field46_soybean_2023",
        "Brazil_DUP2_farm6_field293_wheat_2021",
        "Germany_DUP3_farm1_field39_wheat_2019",
    ]
    loaded = []
    pooled_abs = []
    for fid in examples:
        with np.load(spatial_dir / "maps" / f"{fid}.npz") as archive:
            arrays = {key: archive[key].astype(float) for key in archive.files}
        mask = np.isfinite(arrays["target"])
        anomaly = {}
        for key in ("target", "prediction", "ndvi"):
            anomaly[key] = arrays[key] - np.nanmean(arrays[key])
        pooled_abs.append(np.abs(anomaly["target"][mask]))
        rgb = causal_rgb(fid, mask)
        # Crop every modality with the same mask-derived bounding box so field
        # geometry dominates the panel rather than the surrounding empty raster.
        yy, xx = np.where(mask)
        pad = 2
        y0, y1 = max(0, int(yy.min()) - pad), min(mask.shape[0], int(yy.max()) + pad + 1)
        x0, x1 = max(0, int(xx.min()) - pad), min(mask.shape[1], int(xx.max()) + pad + 1)
        cropped_mask = mask[y0:y1, x0:x1]
        cropped_anomaly = {
            key: np.where(cropped_mask, value[y0:y1, x0:x1], np.nan)
            for key, value in anomaly.items()
        }
        cropped_rgb = rgb[y0:y1, x0:x1].copy()
        cropped_rgb[~cropped_mask] = 1.0
        loaded.append((fid, cropped_mask, cropped_anomaly, cropped_rgb))
    limit = float(np.quantile(np.concatenate(pooled_abs), 0.98))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")

    fig = plt.figure(figsize=(7.16, 4.45), constrained_layout=True)
    gs = fig.add_gridspec(
        4,
        5,
        width_ratios=[1, 1, 1, 1, 1.42],
        height_ratios=[1, 1, 1, 0.075],
        wspace=0.09,
        hspace=0.10,
    )
    map_axes: list[list[plt.Axes]] = []
    image_handle = None
    titles = ["Causal RGB", "Observed", "PhysCrop-Risk", "NDVI baseline"]
    for row_idx, (fid, mask, anomaly, rgb) in enumerate(loaded):
        axes_row = [fig.add_subplot(gs[row_idx, col]) for col in range(4)]
        map_axes.append(axes_row)
        axes_row[0].imshow(rgb, interpolation="nearest")
        for ax, key in zip(axes_row[1:], ("target", "prediction", "ndvi")):
            image_handle = ax.imshow(np.ma.masked_invalid(anomaly[key]), cmap=cmap, norm=norm, interpolation="nearest")
        for col_idx, ax in enumerate(axes_row):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_idx == 0:
                ax.set_title(titles[col_idx], pad=5, fontsize=7.7)
        met = by_id[fid]
        axes_row[0].text(
            -0.06,
            0.5,
            short_field_label(fid, str(met["country"]), str(met["crop"]), float(met["rho_spatial"])),
            transform=axes_row[0].transAxes,
            ha="right",
            va="center",
            fontsize=7.1,
            linespacing=1.25,
        )
    panel_label(map_axes[0][0], "a", x=-0.58, y=1.18)

    cax = fig.add_subplot(gs[3, :4])
    colorbar = fig.colorbar(image_handle, cax=cax, orientation="horizontal")
    colorbar.set_label("Within-field yield anomaly (t ha$^{-1}$)", fontsize=7.2, labelpad=2)
    colorbar.ax.tick_params(labelsize=6.6, length=2, pad=1)
    colorbar.outline.set_linewidth(0.55)

    ax_delta = fig.add_subplot(gs[0, 4])
    clean_axis(ax_delta, grid="x")
    delta_uniform = np.array([float(r["rmse_spatial"]) - float(r["rmse_uniform"]) for r in metrics])
    delta_ndvi = np.array([float(r["rmse_spatial"]) - float(r["rmse_ndvi"]) for r in metrics])
    rng = np.random.default_rng(7441)
    for y, values, color, label in (
        (1, delta_uniform, NAVY, "vs. uniform"),
        (0, delta_ndvi, TEAL, "vs. NDVI"),
    ):
        keep = rng.choice(len(values), min(180, len(values)), replace=False)
        jitter = rng.normal(0, 0.055, len(keep))
        ax_delta.scatter(values[keep], y + jitter, s=5.5, color=color, alpha=0.20, linewidth=0, rasterized=True)
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax_delta.plot([q1, q3], [y, y], color=color, linewidth=5.2, solid_capstyle="butt", zorder=4)
        ax_delta.scatter([median], [y], s=20, color="white", edgecolor=color, linewidth=1.0, zorder=5)
    ax_delta.axvline(0, color=VERMILLION, linestyle="--", linewidth=0.9)
    ax_delta.set_yticks([0, 1], ["vs. NDVI", "vs. uniform"])
    ax_delta.set_xlabel("Paired field RMSE difference")
    ax_delta.set_title("Spatial model lowers field RMSE", loc="left", fontsize=7.9, pad=5)
    panel_label(ax_delta, "b", x=-0.22, y=1.19)

    ax_quality = fig.add_subplot(gs[1:3, 4])
    clean_axis(ax_quality, grid="y")
    rho = np.array([float(r["rho_spatial"]) for r in metrics])
    auc = np.array([float(r["bottom20_auroc"]) for r in metrics])
    parts = ax_quality.violinplot([rho, auc], positions=[0, 1], widths=0.68, showextrema=False)
    for body, color in zip(parts["bodies"], (NAVY, TEAL)):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.18)
        body.set_linewidth(0.8)
    for x_pos, values, color in ((0, rho, NAVY), (1, auc, TEAL)):
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax_quality.plot([x_pos, x_pos], [q1, q3], color=color, linewidth=5.5, solid_capstyle="butt")
        ax_quality.scatter([x_pos], [median], s=25, color="white", edgecolor=color, linewidth=1.1, zorder=4)
    ax_quality.plot([-0.34, 0.34], [0.20, 0.20], "--", color=NAVY, linewidth=0.9, alpha=0.8)
    ax_quality.plot([0.66, 1.34], [0.65, 0.65], "--", color=TEAL, linewidth=0.9, alpha=0.8)
    ax_quality.set_xticks([0, 1], ["Spearman\nρ", "Bottom-20%\nAUROC"])
    ax_quality.tick_params(axis="x", labelsize=6.5)
    ax_quality.set_ylim(-0.2, 1.13)
    ax_quality.text(0.08, 1.025, f"median\n{np.median(rho):.2f}", ha="center", va="bottom", color=NAVY, fontsize=6.5, linespacing=1.0)
    ax_quality.text(1.00, 1.025, f"median\n{np.median(auc):.2f}", ha="center", va="bottom", color=TEAL, fontsize=6.5, linespacing=1.0)
    ax_quality.set_ylabel("Field-level metric")
    ax_quality.set_title("Spatial ranking is consistent across fields", loc="left", fontsize=7.9, pad=5)
    panel_label(ax_quality, "c", x=-0.22, y=1.10)

    save_all(fig, "figure4_spatial_evidence")
    plt.close(fig)


def main() -> None:
    setup_style()
    figure3()
    figure4()
    print(f"Saved Figures 3 and 4 to {OUT}")


if __name__ == "__main__":
    main()
