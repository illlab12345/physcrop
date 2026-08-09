"""Build outcome-controlled, field-level YieldSAT sequence caches.

This adapter targets the downloaded flexible-format archive.  It deliberately
keeps final-audit targets out of the development cache.  Generated artifacts are
data, not hand-edited source files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "YieldSAT_raw_data"
ELIGIBILITY = ROOT / "results/protocol_v2_clean/accept_upgrade_v1/yieldsat_external/field_eligibility.csv"
OUT = ROOT / "results/yieldsat_risk_v1/cache"
DATE_RX = re.compile(r"(\d{8})")
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
DYNAMIC = BANDS + ["NDVI", "NDMI", "NDRE", "EVI", "valid_fraction", "cum_gdd", "cum_precip_mm"]
STATIC_NAMES = [
    "soil_cec", "soil_cfvo", "soil_clay", "soil_nitrogen", "soil_phh2o",
    "soil_sand", "soil_silt", "soil_soc", "dem_aspect", "dem_curvature",
    "dem_dem", "dem_slope", "dem_twi", "area_ha", "circularity",
]


def parse_date(value: str):
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(str(value), fmt).date()
        except (TypeError, ValueError):
            pass
    return None


def finite_float(value):
    try:
        ans = float(value)
        return ans if np.isfinite(ans) else np.nan
    except (TypeError, ValueError):
        return np.nan


def farm_group(field_id: str) -> str:
    match = re.match(r"([^_]+)_([^_]+)_(farm[^_]+)_", field_id)
    return "|".join(match.groups()) if match else field_id


def read_weather(path: Path, seed, harvest):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            day = parse_date(row.get("Date", ""))
            temp = finite_float(row.get("Temp_mean"))
            rain = finite_float(row.get("Total_prec"))
            if day is None or not np.isfinite(temp):
                continue
            # ERA5 archive stores Kelvin and metres.
            temp_c = temp - 273.15 if temp > 100 else temp
            rain_mm = max(0.0, rain * 1000.0) if np.isfinite(rain) else 0.0
            if seed is not None and day < seed:
                continue
            if harvest is not None and day > harvest:
                continue
            rows.append((day, float(np.clip(temp_c - 5.0, 0.0, 25.0)), rain_mm))
    rows.sort()
    if not rows:
        return [], np.array([]), np.array([])
    return [x[0] for x in rows], np.cumsum([x[1] for x in rows]), np.cumsum([x[2] for x in rows])


def value_at(days, values, day):
    if not days:
        return np.nan
    pos = np.searchsorted(np.asarray(days, dtype="datetime64[D]"), np.datetime64(day), side="right") - 1
    return float(values[pos]) if pos >= 0 else np.nan


def masked_raster_mean(path: Path, footprint: np.ndarray):
    arr = np.asarray(tifffile.imread(path), dtype=np.float32)
    if arr.ndim == 2:
        vals = arr[footprint]
    elif arr.ndim == 3 and arr.shape[:2] == footprint.shape:
        vals = arr[footprint, :].reshape(-1)
    else:
        return np.nan
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else np.nan


def extract_one(item, include_target: bool):
    field_id = item["field_id"]
    field = RAW / item["country"] / field_id
    meta_path = next(field.glob("metadata-*.json"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    seed = parse_date(meta.get("seeding_date"))
    harvest = parse_date(meta.get("harvesting_date"))
    weather_path = next((field / "weather").glob("*.csv"))
    weather_days, cum_gdd, cum_rain = read_weather(weather_path, seed, harvest)

    s2_paths = sorted((field / "s2_images").glob("*.tif"))
    scl_paths = {DATE_RX.search(x.stem).group(1): x for x in (field / "scl_masks").glob("*.tif") if DATE_RX.search(x.stem)}
    observations = []
    footprint = None
    for path in s2_paths:
        match = DATE_RX.search(path.stem)
        if not match or match.group(1) not in scl_paths:
            continue
        day = parse_date(match.group(1))
        if day is None or (seed and day < seed) or (harvest and day > harvest):
            continue
        scl = np.asarray(tifffile.imread(scl_paths[match.group(1)]))
        inside = scl > 0
        if footprint is None or inside.sum() > footprint.sum():
            footprint = inside.copy()
        valid = np.isin(scl, (4, 5))
        denominator = int(inside.sum())
        if denominator == 0 or int(valid.sum()) < 20:
            continue
        image = np.asarray(tifffile.imread(path), dtype=np.float32)
        if image.ndim != 3 or image.shape[-1] < 12:
            continue
        means = np.mean(image[valid, :12], axis=0) / 10000.0
        b02, b03, b04, b05, b07, b08, b8a, b11 = means[1], means[2], means[3], means[4], means[6], means[7], means[8], means[10]
        ndvi = (b08 - b04) / (b08 + b04 + 1e-6)
        ndmi = (b08 - b11) / (b08 + b11 + 1e-6)
        ndre = (b07 - b05) / (b07 + b05 + 1e-6)
        evi = 2.5 * (b08 - b04) / (b08 + 6.0 * b04 - 7.5 * b02 + 1.0)
        features = np.r_[means, ndvi, ndmi, ndre, evi, valid.sum() / denominator,
                         value_at(weather_days, cum_gdd, day), value_at(weather_days, cum_rain, day)]
        if np.isfinite(features[:17]).all():
            observations.append((day.toordinal(), features.astype(np.float32)))

    if footprint is None or len(observations) < 4:
        return {"field_id": field_id, "error": "fewer_than_four_usable_acquisitions"}

    static = []
    for stem in ("cec", "cfvo", "clay", "nitrogen", "phh2o", "sand", "silt", "soc"):
        matches = list((field / "soil").glob(f"{stem}_*.tif"))
        static.append(masked_raster_mean(matches[0], footprint) if matches else np.nan)
    for stem in ("aspect", "curvature", "dem", "slope", "twi"):
        matches = list((field / "dem").glob(f"{stem}-*.tif"))
        static.append(masked_raster_mean(matches[0], footprint) if matches else np.nan)
    static.extend([finite_float(meta.get("area_calculated")), finite_float(meta.get("circularity"))])

    target = np.nan
    target_pixels = 0
    if include_target:
        target_path = field / "yield_masks/mean_scaled_yield_masked_regional_statistical_outlier.tif"
        values = np.asarray(tifffile.imread(target_path), dtype=np.float32)
        keep = np.isfinite(values) & (values >= 0)
        target_pixels = int(keep.sum())
        if target_pixels:
            target = float(np.mean(values[keep]))

    return {
        "field_id": field_id, "role": item["split"], "country": item["country"],
        "crop": item["crop"], "year": int(item["year"]), "farm_group": farm_group(field_id),
        "quality": str(meta.get("yieldmap_quality", "")), "dates": np.asarray([x[0] for x in observations], dtype=np.int32),
        "dynamic": np.stack([x[1] for x in observations]), "static": np.asarray(static, dtype=np.float32),
        "target": target, "target_pixels": target_pixels,
    }


def pack(records, output: Path):
    max_time = max(len(x["dates"]) for x in records)
    n = len(records)
    dynamic = np.full((n, max_time, len(DYNAMIC)), np.nan, dtype=np.float32)
    dates = np.zeros((n, max_time), dtype=np.int32)
    lengths = np.zeros(n, dtype=np.int16)
    for i, record in enumerate(records):
        length = len(record["dates"])
        lengths[i] = length
        dates[i, :length] = record["dates"]
        dynamic[i, :length] = record["dynamic"]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, field_id=np.asarray([x["field_id"] for x in records]),
        role=np.asarray([x["role"] for x in records]), country=np.asarray([x["country"] for x in records]),
        crop=np.asarray([x["crop"] for x in records]), year=np.asarray([x["year"] for x in records]),
        farm_group=np.asarray([x["farm_group"] for x in records]), quality=np.asarray([x["quality"] for x in records]),
        dates=dates, lengths=lengths, dynamic=dynamic, static=np.stack([x["static"] for x in records]),
        target=np.asarray([x["target"] for x in records], dtype=np.float32),
        target_pixels=np.asarray([x["target_pixels"] for x in records]),
        dynamic_names=np.asarray(DYNAMIC), static_names=np.asarray(STATIC_NAMES), band_names=np.asarray(BANDS),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", default="adapter_reference,locked_external")
    parser.add_argument("--output", default=str(OUT / "development_sequences.npz"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--include-target", action="store_true")
    args = parser.parse_args()
    roles = set(args.roles.split(","))
    if "audit_validation" in roles and args.include_target:
        raise SystemExit("Refusing to read audit_validation targets before the one-shot final command")
    with ELIGIBILITY.open(encoding="utf-8-sig", newline="") as handle:
        items = [x for x in csv.DictReader(handle) if x["split"] in roles]
    records, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_one, item, args.include_target): item for item in items}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                record = future.result()
                (errors if "error" in record else records).append(record)
            except Exception as exc:  # preserve an auditable exclusion rather than abort 40 GB scan
                errors.append({"field_id": futures[future]["field_id"], "error": f"{type(exc).__name__}: {exc}"})
            if done == 1 or done % 50 == 0:
                print(f"processed={done}/{len(items)} eligible={len(records)} excluded={len(errors)}", flush=True)
    records.sort(key=lambda x: x["field_id"])
    pack(records, Path(args.output))
    report = {
        "roles": sorted(roles), "requested_fields": len(items), "cached_fields": len(records),
        "excluded_fields": len(errors), "include_target": bool(args.include_target),
        "audit_targets_read": bool("audit_validation" in roles and args.include_target),
        "dynamic_features": DYNAMIC, "static_features": STATIC_NAMES,
        "output_sha256": hashlib.sha256(Path(args.output).read_bytes()).hexdigest(), "errors": errors,
    }
    report_path = Path(args.output).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "errors"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
