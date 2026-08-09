"""Generate publication-ready system figures for the PhysCrop-Risk paper.

Figure 2 is intentionally self-contained and uses only vector primitives so the
SVG remains fully editable in Inkscape, Illustrator, and recent PowerPoint.
"""
from __future__ import annotations

import csv
from datetime import date, datetime
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "论文"

NAVY = "#244F73"
TEAL = "#1B9E77"
ORANGE = "#D98900"
VERMILLION = "#C84C36"
PURPLE = "#7A5195"
GRAY = "#737B83"
LIGHT_GRAY = "#D9DEE3"
PALE_BLUE = "#EDF3F7"
PALE_TEAL = "#EAF6F2"
PALE_ORANGE = "#FBF2E2"
PALE_RED = "#F9ECE9"
INK = "#20262E"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_all(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}.png", dpi=450, bbox_inches="tight", pad_inches=0.02)


def rounded(ax, xy, width, height, *, fc="white", ec=LIGHT_GRAY, lw=0.8, radius=0.012, zorder=1):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        transform=ax.transAxes,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=GRAY, lw=1.15, zorder=3):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=lw,
            color=color,
            transform=ax.transAxes,
            shrinkA=1,
            shrinkB=1,
            zorder=zorder,
        )
    )


def tree(ax, center, scale, color):
    """Small decision-tree glyph in axes coordinates."""
    x, y = center
    pts = [
        (x, y + 0.055 * scale),
        (x - 0.045 * scale, y),
        (x + 0.045 * scale, y),
        (x - 0.066 * scale, y - 0.052 * scale),
        (x - 0.024 * scale, y - 0.052 * scale),
        (x + 0.024 * scale, y - 0.052 * scale),
        (x + 0.066 * scale, y - 0.052 * scale),
    ]
    for a, b in ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)):
        ax.plot(
            [pts[a][0], pts[b][0]],
            [pts[a][1], pts[b][1]],
            color=color,
            linewidth=0.75,
            transform=ax.transAxes,
            zorder=5,
        )
    for i, (px, py) in enumerate(pts):
        ax.add_patch(
            Circle(
                (px, py),
                0.0095 * scale,
                transform=ax.transAxes,
                facecolor="white" if i < 3 else color,
                edgecolor=color,
                linewidth=0.75,
                zorder=6,
            )
        )


def stage_header(ax, x0, width, number, title, subtitle, color):
    ax.add_patch(Circle((x0 + 0.019, 0.722), 0.014, transform=ax.transAxes,
                        facecolor=color, edgecolor="none", zorder=5))
    ax.text(x0 + 0.019, 0.722, str(number), transform=ax.transAxes, ha="center",
            va="center", color="white", fontsize=6.5, fontweight="bold", zorder=6)
    ax.text(x0 + 0.041, 0.735, title, transform=ax.transAxes, ha="left", va="center",
            fontsize=8.2, fontweight="bold", color=INK)
    ax.text(x0 + 0.041, 0.706, subtitle, transform=ax.transAxes, ha="left", va="center",
            fontsize=6.1, color=GRAY)


def figure2() -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(7.16, 3.55))
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Phenological monitoring axis: observations only become available as the
    # season unfolds. This encodes causality before the processing stages.
    ax.text(0.018, 0.948, "CAUSAL PARTIAL-SEASON MONITORING", transform=ax.transAxes,
            fontsize=6.6, color=NAVY, fontweight="bold", va="center")
    ax.plot([0.375, 0.962], [0.948, 0.948], transform=ax.transAxes,
            color=LIGHT_GRAY, linewidth=1.1, zorder=1)
    look_x = [0.39, 0.585, 0.78]
    look_labels = ["Look 1", "Look 2", "Look 3"]
    for i, (lx, label) in enumerate(zip(look_x, look_labels), start=1):
        ax.add_patch(Circle((lx, 0.948), 0.009, transform=ax.transAxes,
                            facecolor=NAVY, edgecolor="white", linewidth=0.7, zorder=3))
        ax.text(lx, 0.915, label, transform=ax.transAxes, ha="center", va="top",
                fontsize=6.4, color=INK)
    ax.text(0.966, 0.948, "GDD", transform=ax.transAxes, ha="left", va="center",
            fontsize=6.2, color=GRAY)
    ax.text(0.982, 0.888, "No future imagery or weather", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.0, color=GRAY, style="italic")

    # Five-stage layout. Soft grouping bands preserve structure without turning
    # the pipeline into a dashboard of disconnected cards.
    xs = [0.018, 0.205, 0.402, 0.575, 0.787]
    ws = [0.158, 0.166, 0.142, 0.181, 0.195]
    fills = [PALE_BLUE, PALE_TEAL, PALE_BLUE, PALE_ORANGE, PALE_RED]
    colors = [NAVY, TEAL, NAVY, ORANGE, VERMILLION]
    for x0, w, fc, color in zip(xs, ws, fills, colors):
        rounded(ax, (x0, 0.205), w, 0.575, fc=fc, ec=color, lw=0.65, radius=0.013)

    for i in range(4):
        arrow(ax, (xs[i] + ws[i] + 0.005, 0.49), (xs[i + 1] - 0.006, 0.49),
              color=GRAY, lw=1.1, zorder=8)

    stage_header(ax, xs[0], ws[0], 1, "Observe", "causal inputs", NAVY)
    stage_header(ax, xs[1], ws[1], 2, "Align", "GDD features", TEAL)
    stage_header(ax, xs[2], ws[2], 3, "Predict", "dual trees", NAVY)
    stage_header(ax, xs[3], ws[3], 4, "Calibrate", "one-sided bounds", ORANGE)
    stage_header(ax, xs[4], ws[4], 5, "Act", "persistent alerts", VERMILLION)

    # Stage 1: observation stack with two image chips, a weather trace, and soil.
    x0 = xs[0]
    chip_x, chip_y, chip_w, chip_h = x0 + 0.021, 0.484, 0.054, 0.142
    for k, (dx, dy) in enumerate(((0.015, 0.020), (0.007, 0.010), (0, 0))):
        ax.add_patch(Rectangle((chip_x + dx, chip_y + dy), chip_w, chip_h,
                               transform=ax.transAxes, facecolor=["#B7C98A", "#89AF70", "#588C5A"][k],
                               edgecolor="white", linewidth=0.65, zorder=4 + k))
        # Crop-row texture.
        for t in np.linspace(0.008, chip_w - 0.006, 4):
            ax.plot([chip_x + dx + t, chip_x + dx + t - 0.015],
                    [chip_y + dy + 0.006, chip_y + dy + chip_h - 0.006],
                    transform=ax.transAxes, color="#E3E7B1", linewidth=0.45,
                    alpha=0.75, zorder=5 + k)
    ax.text(chip_x + chip_w / 2, 0.462, "Sentinel-2", transform=ax.transAxes,
            ha="center", va="top", fontsize=6.0, color=INK)

    wx = np.linspace(x0 + 0.093, x0 + 0.141, 7)
    wy = np.array([0.545, 0.562, 0.535, 0.585, 0.575, 0.608, 0.592])
    ax.plot(wx, wy, transform=ax.transAxes, color=NAVY, linewidth=1.0, zorder=5)
    ax.fill_between(wx, 0.515, wy, transform=ax.transAxes, color=NAVY, alpha=0.10, zorder=4)
    ax.plot([x0 + 0.093, x0 + 0.141], [0.515, 0.515], transform=ax.transAxes,
            color=GRAY, linewidth=0.5)
    ax.text(x0 + 0.117, 0.492, "ERA5", transform=ax.transAxes,
            ha="center", va="top", fontsize=6.0)
    soil_y = [0.409, 0.381, 0.353]
    soil_c = ["#C9A66B", "#A77C47", "#805B35"]
    for sy, sc in zip(soil_y, soil_c):
        ax.add_patch(Rectangle((x0 + 0.031, sy), 0.096, 0.024, transform=ax.transAxes,
                               facecolor=sc, edgecolor="white", linewidth=0.4))
    ax.text(x0 + 0.079, 0.331, "soil · terrain", transform=ax.transAxes,
            ha="center", va="top", fontsize=5.9, color=INK)
    ax.text(x0 + ws[0] / 2, 0.283, "field geometry", transform=ax.transAxes,
            ha="center", va="center", fontsize=5.9, color=INK)
    ax.text(x0 + ws[0] / 2, 0.247, "dynamic + static", transform=ax.transAxes,
            ha="center", va="center", fontsize=5.8, color=GRAY)

    # Stage 2: GDD-aligned feature tensor.
    x0 = xs[1]
    gx, gy, gw, gh = x0 + 0.026, 0.365, 0.112, 0.255
    vals = np.array([
        [0.18, 0.30, 0.44, 0.60, 0.70, 0.78],
        [0.26, 0.42, 0.57, 0.75, 0.68, 0.55],
        [0.72, 0.66, 0.58, 0.48, 0.37, 0.30],
        [0.38, 0.48, 0.54, 0.62, 0.73, 0.82],
        [0.62, 0.57, 0.49, 0.45, 0.40, 0.34],
    ])
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("feature", [PALE_TEAL, "#72B9A4", TEAL])
    ax.imshow(vals, extent=(gx, gx + gw, gy, gy + gh), origin="lower", cmap=cmap,
              interpolation="nearest", aspect="auto", transform=ax.transAxes, zorder=4)
    for k in range(1, 6):
        ax.plot([gx + gw * k / 6] * 2, [gy, gy + gh], transform=ax.transAxes,
                color="white", linewidth=0.45, alpha=0.85, zorder=5)
    for k in range(1, 5):
        ax.plot([gx, gx + gw], [gy + gh * k / 5] * 2, transform=ax.transAxes,
                color="white", linewidth=0.45, alpha=0.85, zorder=5)
    ax.plot([gx + gw * 0.72] * 2, [gy - 0.012, gy + gh + 0.012], transform=ax.transAxes,
            color=ORANGE, linewidth=1.35, zorder=6)
    ax.text(gx + gw * 0.72, gy + gh + 0.024, "look t", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=5.7, color=ORANGE, fontweight="bold")
    ax.text(gx + gw / 2, 0.335, "features × GDD", transform=ax.transAxes,
            ha="center", va="top", fontsize=5.9)
    ax.text(x0 + ws[1] / 2, 0.283, "12 bands + 4 indices", transform=ax.transAxes,
            ha="center", fontsize=5.7, color=INK)
    ax.text(x0 + ws[1] / 2, 0.239, "weather · soil · terrain", transform=ax.transAxes,
            ha="center", fontsize=5.5, color=GRAY)

    # Stage 3: two complementary tree ensembles and frozen weighted fusion.
    x0 = xs[2]
    tree(ax, (x0 + 0.046, 0.534), 0.75, NAVY)
    tree(ax, (x0 + 0.098, 0.534), 0.75, TEAL)
    ax.text(x0 + 0.046, 0.430, "ET", transform=ax.transAxes, ha="center",
            fontsize=5.8, color=NAVY, fontweight="bold")
    ax.text(x0 + 0.098, 0.430, "HGB", transform=ax.transAxes, ha="center",
            fontsize=5.8, color=TEAL, fontweight="bold")
    bx, by, bw = x0 + 0.026, 0.375, 0.093
    ax.add_patch(Rectangle((bx, by), bw * 0.9, 0.018, transform=ax.transAxes,
                           facecolor=NAVY, edgecolor="none", zorder=4))
    ax.add_patch(Rectangle((bx + bw * 0.9, by), bw * 0.1, 0.018, transform=ax.transAxes,
                           facecolor=TEAL, edgecolor="none", zorder=4))
    ax.text(x0 + ws[2] / 2, 0.348, "90% / 10% fusion", transform=ax.transAxes,
            ha="center", va="top", fontsize=5.9, color=GRAY)
    rounded(ax, (x0 + 0.038, 0.265), 0.067, 0.051, fc="white", ec=NAVY, lw=0.8, radius=0.01, zorder=4)
    ax.text(x0 + 0.0715, 0.291, r"$\hat{y}_t$", transform=ax.transAxes,
            ha="center", va="center", fontsize=11, color=NAVY, fontweight="bold", zorder=5)

    # Stage 4: one-sided conformal upper bound and fallback hierarchy.
    x0 = xs[3]
    px0, px1, py0, py1 = x0 + 0.028, x0 + 0.155, 0.397, 0.622
    ax.plot([px0, px0], [py0, py1], transform=ax.transAxes, color=GRAY, linewidth=0.6)
    ax.plot([px0, px1], [py0, py0], transform=ax.transAxes, color=GRAY, linewidth=0.6)
    threshold_y = 0.482
    pred_y, upper_y = 0.444, 0.552
    ax.plot([px0, px1], [threshold_y, threshold_y], transform=ax.transAxes,
            color=VERMILLION, linewidth=0.9, linestyle=(0, (3, 2)))
    ax.text(px0 + 0.004, threshold_y + 0.010, r"low-yield $\tau_g$",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=4.9, color=VERMILLION)
    ax.scatter([x0 + 0.088], [pred_y], transform=ax.transAxes, s=18,
               facecolor=NAVY, edgecolor="white", linewidth=0.5, zorder=6)
    ax.plot([x0 + 0.088, x0 + 0.088], [pred_y, upper_y], transform=ax.transAxes,
            color=ORANGE, linewidth=1.7, zorder=5)
    ax.plot([x0 + 0.078, x0 + 0.098], [upper_y, upper_y], transform=ax.transAxes,
            color=ORANGE, linewidth=1.7, zorder=5)
    ax.text(x0 + 0.103, pred_y, r"$\hat{y}_t$", transform=ax.transAxes,
            va="center", fontsize=6.2, color=NAVY)
    ax.text(x0 + 0.103, upper_y, r"$U_t$", transform=ax.transAxes,
            va="center", fontsize=6.2, color=ORANGE, fontweight="bold")
    ax.text(x0 + ws[3] / 2, 0.354, r"$U_t=\hat{y}_t+q_{1-\alpha_t}$",
            transform=ax.transAxes, ha="center", va="center", fontsize=7.0, color=INK)
    ax.text(x0 + ws[3] / 2, 0.307, "country–crop → crop",
            transform=ax.transAxes, ha="center", fontsize=5.3, color=GRAY)
    ax.text(x0 + ws[3] / 2, 0.278, "→ global fallback",
            transform=ax.transAxes, ha="center", fontsize=5.3, color=GRAY)
    ax.text(x0 + ws[3] / 2, 0.238, r"$\alpha$ spending across looks",
            transform=ax.transAxes, ha="center", fontsize=5.5, color=ORANGE)

    # Stage 5: persistence logic and 10 m spatial evidence.
    x0 = xs[4]
    node_x = [x0 + 0.031, x0 + 0.087, x0 + 0.143]
    for j in range(2):
        arrow(ax, (node_x[j] + 0.016, 0.582), (node_x[j + 1] - 0.016, 0.582),
              color=GRAY, lw=0.85, zorder=5)
    for nx, fill, edge, label, mark in zip(
        node_x,
        ["white", PALE_ORANGE, PALE_RED],
        [GRAY, ORANGE, VERMILLION],
        ["L1", "L2", "L3"],
        ["—", "watch", "action"],
    ):
        ax.add_patch(Circle((nx, 0.582), 0.019, transform=ax.transAxes,
                            facecolor=fill, edgecolor=edge, linewidth=1.0, zorder=6))
        ax.text(nx, 0.582, label, transform=ax.transAxes, ha="center", va="center",
                fontsize=5.7, color=edge, fontweight="bold", zorder=7)
        ax.text(nx, 0.543, mark, transform=ax.transAxes, ha="center", va="top",
                fontsize=5.7, color=edge)
    ax.text(x0 + ws[4] / 2, 0.497, "two consecutive", transform=ax.transAxes,
            ha="center", fontsize=5.8, color=INK)
    ax.text(x0 + ws[4] / 2, 0.466, "bound crossings", transform=ax.transAxes,
            ha="center", fontsize=5.8, color=INK)

    # Spatial evidence grid: neutral, high-risk, and transition cells.
    grid = np.array([
        [0, 0, 1, 1, 1, 0],
        [0, 1, 1, 2, 2, 0],
        [0, 1, 2, 2, 2, 1],
        [0, 1, 2, 2, 1, 1],
        [0, 0, 1, 1, 1, 0],
    ])
    palette = ["#DDE8D8", "#E6B47A", VERMILLION]
    sx, sy, cell = x0 + 0.058, 0.294, 0.016
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            ax.add_patch(Rectangle((sx + c * cell, sy + r * cell), cell, cell,
                                   transform=ax.transAxes, facecolor=palette[grid[r, c]],
                                   edgecolor="white", linewidth=0.45, zorder=5))
    outline = Polygon(
        [(sx - 0.006, sy + 0.010), (sx + 0.014, sy - 0.008),
         (sx + 0.092, sy - 0.004), (sx + 0.106, sy + 0.036),
         (sx + 0.091, sy + 0.086), (sx + 0.016, sy + 0.092)],
        closed=True, transform=ax.transAxes, fill=False, edgecolor=INK,
        linewidth=0.75, zorder=7,
    )
    ax.add_patch(outline)
    ax.text(x0 + ws[4] / 2, 0.258, "10 m within-field evidence",
            transform=ax.transAxes, ha="center", fontsize=5.9, color=GRAY)

    # A compact synthesis line closes the method without duplicating the blocks.
    ax.plot([0.018, 0.982], [0.154, 0.154], transform=ax.transAxes,
            color=LIGHT_GRAY, linewidth=0.75)
    footer = [
        (0.084, "causal inputs", NAVY),
        (0.282, "phenology alignment", TEAL),
        (0.487, "point prediction", NAVY),
        (0.677, "calibrated risk", ORANGE),
        (0.895, "actionable evidence", VERMILLION),
    ]
    for fx, label, color in footer:
        ax.add_patch(Circle((fx - 0.048, 0.104), 0.006, transform=ax.transAxes,
                            facecolor=color, edgecolor="none"))
        ax.text(fx - 0.036, 0.104, label, transform=ax.transAxes,
                ha="left", va="center", fontsize=6.2, color=INK)

    save_all(fig, "figure2_physcrop_risk_framework")
    plt.close(fig)


def _audit_case_rows(field_id: str) -> list[dict[str, str]]:
    path = ROOT / "results" / "yieldsat_risk_v1" / "final_audit" / "final_predictions.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["field_id"] == field_id]
    rows.sort(key=lambda row: int(row["look"]))
    if len(rows) != 3:
        raise RuntimeError(f"Expected three looks for {field_id}, found {len(rows)}")
    return rows


def _audit_rgb_sequence(field_id: str) -> tuple[list[np.ndarray], list[str], list[float]]:
    """Return the latest sufficiently clear causal RGB at each frozen GDD look."""
    results = ROOT / "results" / "yieldsat_risk_v1"
    cache = np.load(results / "cache" / "audit_features_blind.npz", allow_pickle=False)
    ids = cache["field_id"].astype(str)
    matches = np.flatnonzero(ids == field_id)
    if len(matches) != 1:
        cache.close()
        raise RuntimeError(f"Cannot uniquely locate audit field {field_id}")
    idx = int(matches[0])
    country = str(cache["country"][idx])
    crop = str(cache["crop"][idx])
    length = int(cache["lengths"][idx])
    ordinals = cache["dates"][idx, :length]
    gdd = cache["dynamic"][idx, :length, 17]
    folder = ROOT / "YieldSAT_raw_data" / country / field_id
    target = np.asarray(
        tifffile.imread(folder / "yield_masks" / "mean_scaled_yield_masked_regional_statistical_outlier.tif"),
        dtype=float,
    )
    field_mask = np.isfinite(target) & (target >= 0)

    images: list[np.ndarray] = []
    labels: list[str] = []
    used_gdd: list[float] = []
    for look in (1, 2, 3):
        bundle = joblib.load(results / "experiment_02_early_curve" / f"model_look{look}.joblib")
        cutoff = float(bundle["look_gdd"][crop])
        positions = np.flatnonzero(np.isfinite(gdd) & (gdd <= cutoff + 1e-6))
        chosen = None
        for pos in positions[::-1]:
            stamp = date.fromordinal(int(ordinals[pos])).strftime("%Y%m%d")
            s2_path = folder / "s2_images" / f"S2_L2A_{stamp}.tif"
            scl_path = folder / "scl_masks" / f"S2_L2A_SCL_{stamp}.tif"
            if not s2_path.exists() or not scl_path.exists():
                continue
            scl = np.asarray(tifffile.imread(scl_path))
            valid = np.isin(scl, (4, 5)) & field_mask
            if valid.sum() / max(field_mask.sum(), 1) < 0.35:
                continue
            s2 = np.asarray(tifffile.imread(s2_path), dtype=np.float32) / 10000.0
            chosen = (s2, valid, stamp, float(gdd[pos]))
            break
        if chosen is None:
            cache.close()
            raise RuntimeError(f"No clear causal image for look {look}: {field_id}")
        s2, valid, stamp, actual_gdd = chosen
        rgb = s2[:, :, [3, 2, 1]]
        values = rgb[valid]
        lo, hi = np.nanpercentile(values, [2, 98])
        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.86
        # Retain cloudy holes as a neutral gray and remove non-field context.
        rgb[field_mask & ~valid] = 0.88
        rgb[~field_mask] = 1.0
        images.append(rgb)
        labels.append(datetime.strptime(stamp, "%Y%m%d").strftime("%d %b %Y"))
        used_gdd.append(actual_gdd)
    cache.close()
    return images, labels, used_gdd


def _clean_plot(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRAY)
    ax.spines["bottom"].set_color(GRAY)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.tick_params(length=2.5, width=0.6, colors=INK, labelsize=6.4)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.55, alpha=0.75, zorder=0)


def figure1() -> None:
    """Create a data-grounded teaser from a true-positive held-out audit field."""
    setup_style()
    results = ROOT / "results" / "yieldsat_risk_v1"
    field_id = "Argentina_DUP1_farm45_field633_soybean_2023"
    rows = _audit_case_rows(field_id)
    images, dates, gdd = _audit_rgb_sequence(field_id)
    report = json.loads((results / "final_audit" / "final_report.json").read_text(encoding="utf-8"))

    pvals = np.array([float(row["pvalue"]) for row in rows])
    alphas = np.array([float(row["alpha"]) for row in rows])
    target = float(rows[-1]["target"])
    cutoff = float(rows[-1]["cutoff"])
    prediction = float(rows[-1]["prediction"])
    late = report["looks"]["3"]["metrics"]
    alert = report["action_alert"]

    fig = plt.figure(figsize=(7.16, 4.35), facecolor="white")
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis("off")

    # Outcome-led headline: this figure is the paper's visual abstract, while
    # Figure 2 carries the full method specification.
    canvas.text(0.035, 0.952, "Causal evidence → calibrated persistence → action before harvest",
                fontsize=12.2, fontweight="bold", color=INK, ha="left", va="center")
    canvas.text(0.035, 0.906, "Outcome-blind audit example · Argentina soybean · 2023 season",
                fontsize=6.8, color=GRAY, ha="left", va="center")
    canvas.text(0.965, 0.906, "Only observations available by each GDD look are used",
                fontsize=6.5, color=NAVY, ha="right", va="center", style="italic")

    canvas.text(0.018, 0.846, "a", fontsize=10, fontweight="bold", va="top")
    canvas.text(0.045, 0.846, "Partial-season Sentinel-2 evidence", fontsize=8.0,
                fontweight="bold", va="top")
    canvas.plot([0.063, 0.545], [0.803, 0.803], color=LIGHT_GRAY, linewidth=1.0)

    image_lefts = [0.052, 0.225, 0.398]
    status = [("WATCH", ORANGE), ("ACTION", VERMILLION), ("MAINTAIN", VERMILLION)]
    for i, (left, rgb, day, thermal, (state, color)) in enumerate(
        zip(image_lefts, images, dates, gdd, status), start=1
    ):
        center = left + 0.0725
        canvas.add_patch(Circle((center, 0.803), 0.007, facecolor=NAVY,
                                edgecolor="white", linewidth=0.55, zorder=4))
        img_ax = fig.add_axes([left, 0.535, 0.145, 0.235])
        img_ax.imshow(rgb, interpolation="nearest")
        img_ax.contour(np.any(rgb < 0.995, axis=2), levels=[0.5], colors=["white"],
                       linewidths=0.45, alpha=0.9)
        img_ax.set_xticks([])
        img_ax.set_yticks([])
        for spine in img_ax.spines.values():
            spine.set_color("white")
            spine.set_linewidth(0.8)
        canvas.text(center, 0.785, f"Look {i} · {thermal:.0f} GDD", fontsize=6.5,
                    ha="center", va="top", color=INK, fontweight="bold")
        canvas.text(center, 0.512, day, fontsize=5.9, ha="center", va="top", color=GRAY)
        rounded(canvas, (center - 0.034, 0.463), 0.068, 0.030, fc="white", ec=color,
                lw=0.75, radius=0.008, zorder=4)
        canvas.text(center, 0.478, state, fontsize=5.7, ha="center", va="center",
                    color=color, fontweight="bold", zorder=5)
    canvas.text(0.314, 0.437, "Two consecutive risk hits trigger action at Look 2",
                ha="center", va="center", fontsize=6.1, color=INK)

    # Calibrated evidence plot.
    canvas.text(0.595, 0.846, "b", fontsize=10, fontweight="bold", va="top")
    canvas.text(0.622, 0.846, "Frozen sequential risk evidence", fontsize=8.0,
                fontweight="bold", va="top")
    risk_ax = fig.add_axes([0.628, 0.535, 0.323, 0.235])
    x = np.arange(1, 4)
    risk_ax.fill_between(x, 0.0035, alphas, color=PALE_RED, alpha=0.9, zorder=1)
    risk_ax.plot(x, alphas, "--s", color=ORANGE, linewidth=1.35, markersize=3.8,
                 label=r"spending boundary $\alpha_t$", zorder=3)
    risk_ax.plot(x, pvals, "-o", color=NAVY, linewidth=1.75, markersize=4.4,
                 label=r"conformal evidence $p_t$", zorder=4)
    for xi, value in zip(x, pvals):
        risk_ax.text(xi, value * 0.86, f"{value:.4f}", color=NAVY, fontsize=5.8,
                     ha="center", va="top")
    risk_ax.set_yscale("log")
    risk_ax.set_xlim(0.76, 3.55)
    risk_ax.set_ylim(0.0034, 0.035)
    risk_ax.set_xticks(x, ["Look 1", "Look 2", "Look 3"])
    risk_ax.set_yticks([0.005, 0.01, 0.02, 0.03], ["0.005", "0.010", "0.020", "0.030"])
    risk_ax.set_ylabel("Evidence level", fontsize=6.7)
    _clean_plot(risk_ax)
    risk_ax.text(3.10, alphas[-1] * 1.08, r"$\alpha_t$", color=ORANGE,
                 fontsize=6.2, va="bottom")
    risk_ax.text(3.20, pvals[-1] * 1.16, r"$p_t$", color=NAVY,
                 fontsize=6.2, va="bottom")
    risk_ax.text(0.82, 0.0315, r"all looks: $p_t \leq \alpha_t$",
                 fontsize=5.8, color=VERMILLION, ha="left", va="top")

    # Bottom synthesis: sequence-level decision, observed outcome, and frozen
    # cohort metrics. Dividers replace card chrome to keep a paper-like rhythm.
    canvas.plot([0.035, 0.965], [0.397, 0.397], color=LIGHT_GRAY, linewidth=0.8)
    canvas.text(0.018, 0.370, "c", fontsize=10, fontweight="bold", va="top")
    canvas.text(0.045, 0.370, "Decision and held-out outcome", fontsize=8.0,
                fontweight="bold", va="top")

    seq_x = [0.082, 0.175, 0.268]
    seq_labels = [("WATCH", ORANGE), ("ACTION", VERMILLION), ("MAINTAIN", VERMILLION)]
    for j in range(2):
        arrow(canvas, (seq_x[j] + 0.024, 0.270), (seq_x[j + 1] - 0.024, 0.270),
              color=GRAY, lw=0.9, zorder=3)
    for sx, (label, color) in zip(seq_x, seq_labels):
        canvas.add_patch(Circle((sx, 0.270), 0.020, facecolor="white", edgecolor=color,
                                linewidth=1.3, zorder=4))
        canvas.add_patch(Circle((sx, 0.270), 0.007, facecolor=color, edgecolor="none", zorder=5))
        canvas.text(sx, 0.231, label, ha="center", va="top", fontsize=5.8,
                    fontweight="bold", color=color)
    canvas.text(0.175, 0.175, "Two-hit persistence filters isolated excursions",
                ha="center", va="center", fontsize=5.5, color=GRAY)

    canvas.plot([0.350, 0.350], [0.105, 0.345], color=LIGHT_GRAY, linewidth=0.75)
    canvas.text(0.382, 0.327, "Harvest confirmation", fontsize=6.8, fontweight="bold",
                ha="left", va="center")
    yield_ax = fig.add_axes([0.387, 0.165, 0.277, 0.118])
    yield_ax.set_xlim(0, 3.0)
    yield_ax.set_ylim(-0.45, 0.50)
    yield_ax.axvline(cutoff, color=ORANGE, linestyle=(0, (3, 2)), linewidth=1.15)
    yield_ax.scatter([target], [0.20], s=35, color=VERMILLION, edgecolor="white",
                     linewidth=0.6, zorder=4)
    yield_ax.scatter([prediction], [-0.10], s=31, color=NAVY, marker="D", edgecolor="white",
                     linewidth=0.6, zorder=4)
    yield_ax.text(target, 0.35, f"observed {target:.2f}", color=VERMILLION,
                  fontsize=5.8, ha="center", va="bottom")
    yield_ax.text(prediction, -0.25, f"Look 3 prediction {prediction:.2f}", color=NAVY,
                  fontsize=5.7, ha="center", va="top")
    yield_ax.text(cutoff - 0.03, 0.44, f"low-yield cutoff {cutoff:.2f}", color=ORANGE,
                  fontsize=5.7, ha="right", va="bottom")
    yield_ax.set_xlabel("Yield (t ha$^{-1}$)", fontsize=6.3, labelpad=1)
    yield_ax.set_yticks([])
    yield_ax.set_xticks([0, 1, 2, 3])
    yield_ax.tick_params(axis="x", labelsize=5.8, length=2.2, width=0.55)
    yield_ax.spines["top"].set_visible(False)
    yield_ax.spines["right"].set_visible(False)
    yield_ax.spines["left"].set_visible(False)
    yield_ax.spines["bottom"].set_color(GRAY)
    canvas.plot([0.694, 0.694], [0.105, 0.345], color=LIGHT_GRAY, linewidth=0.75)
    canvas.text(0.726, 0.327, f"Frozen audit · n = {report['audit_fields']}",
                fontsize=6.8, fontweight="bold", ha="left", va="center")
    metric_lines = [
        ("Look 3 prediction", f"R² {late['r2']:.3f}", NAVY),
        ("Low-yield ranking", f"AUROC {late['auroc']:.3f}", TEAL),
        ("Persistent action", f"FPR {100*alert['fpr']:.2f}%", VERMILLION),
        ("Alert reliability", f"Precision {100*alert['precision']:.1f}%", ORANGE),
    ]
    yy = 0.278
    for label, value, color in metric_lines:
        canvas.add_patch(Circle((0.728, yy), 0.005, facecolor=color, edgecolor="none"))
        canvas.text(0.741, yy, label, fontsize=5.9, color=GRAY, ha="left", va="center")
        canvas.text(0.952, yy, value, fontsize=6.2, color=INK, ha="right", va="center",
                    fontweight="bold")
        yy -= 0.048

    save_all(fig, "figure1_heldout_alert_overview")
    plt.close(fig)


if __name__ == "__main__":
    figure2()
    figure1()
