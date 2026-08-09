"""Create Look 1/2/3 RGB previews for one YieldSAT field.

Usage: python scripts/make_look_example.py <field_id> [output_dir]
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
P0A = ROOT / "results/yieldsat_risk_v1/post_audit_p0p1/p0a_lead_times.csv"
VALID_SCL = {4, 5}
LOOK_LABELS = {1: "Look 1 · 35% GDD", 2: "Look 2 · 50% GDD", 3: "Look 3 · 65% GDD"}


def read_csv(path: Path) -> list[dict[str, str]]:
    import csv
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def percentile_stretch(channel: np.ndarray) -> np.ndarray:
    finite = channel[np.isfinite(channel)]
    if finite.size == 0:
        return np.zeros_like(channel)
    low, high = np.percentile(finite, (2, 98))
    if high <= low:
        return np.zeros_like(channel)
    return np.clip((channel - low) / (high - low), 0, 1)


def render_rgb(image: np.ndarray, scl: np.ndarray, field_mask: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] <= 16:
        bands = image
        h, w = image.shape[1], image.shape[2]
    else:
        bands = np.moveaxis(image, -1, 0)
        h, w = image.shape[0], image.shape[1]

    def align2d(value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value)
        if arr.shape == (h, w):
            return arr
        if arr.shape == (w, h):
            return arr.T
        raise ValueError(f"spatial mismatch: image=({h},{w}) mask={arr.shape}")

    scl = align2d(scl)
    field_mask = align2d(field_mask)
    # Band order: B01..B12 (see build_yieldsat_risk_cache.BANDS)
    red = percentile_stretch(bands[3].astype(np.float32))
    green = percentile_stretch(bands[2].astype(np.float32))
    blue = percentile_stretch(bands[1].astype(np.float32))
    rgb = np.stack([red, green, blue], axis=-1)
    valid = np.isin(scl, list(VALID_SCL)) & field_mask
    # Outside the field: dark gray backdrop; clouds/invalid pixels: light gray
    outside = np.zeros_like(rgb)
    outside[:] = 0.18
    invalid_cloud = np.zeros_like(rgb)
    invalid_cloud[:] = 0.72
    rgb = np.where(valid[..., None], rgb, np.where(field_mask[..., None], invalid_cloud, outside))
    return rgb


def main() -> None:
    field_id = sys.argv[1]
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "demo_look_example" / field_id
    rows = {int(row["look"]): row for row in read_csv(P0A) if row["field_id"] == field_id}
    if len(rows) != 3:
        raise SystemExit(f"Expected 3 looks for {field_id}, got {len(rows)}")
    field_dir = ROOT / "YieldSAT_raw_data" / field_id.split("_")[0] / field_id
    s2_dir = field_dir / "s2_images"
    scl_dir = field_dir / "scl_masks"
    yield_path = field_dir / "yield_masks/mean_scaled_yield_masked_regional_statistical_outlier.tif"
    yield_map = np.asarray(tifffile.imread(yield_path), dtype=np.float32)
    field_mask = np.isfinite(yield_map) & (yield_map >= 0)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for look in (1, 2, 3):
        date = rows[look]["last_acquisition"]
        s2_path = s2_dir / f"S2_L2A_{date.replace('-', '')}.tif"
        scl_path = scl_dir / f"S2_L2A_SCL_{date.replace('-', '')}.tif"
        if not s2_path.exists() or not scl_path.exists():
            raise SystemExit(f"Missing files for look {look}: {s2_path} / {scl_path}")
        image = np.asarray(tifffile.imread(s2_path), dtype=np.float32)
        scl = np.asarray(tifffile.imread(scl_path), dtype=np.int16)
        rgb = render_rgb(image, scl, field_mask)
        img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        banner = Image.new("RGB", (img.width, img.height + 42), (20, 44, 32))
        banner.paste(img, (0, 42))
        draw = ImageDraw.Draw(banner)
        draw.text((14, 10), f"{LOOK_LABELS[look]}  ·  {date}  ·  {field_id}", fill=(255, 255, 255))
        out_path = out_dir / f"look{look}_35_50_65gdd.png"
        banner.save(out_path)
        written.append(str(out_path))
    print("\n".join(written))


if __name__ == "__main__":
    main()
