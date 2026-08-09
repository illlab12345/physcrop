"""Outcome-blind feasibility audit for YieldSAT 64 px chip placement."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "YieldSAT_raw_data"
OUT = ROOT / "results/protocol_v2_clean/accept_upgrade_v1/yieldsat_external"


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def positions(length: int, size: int = 64, stride: int = 8):
    if length <= size:
        return [0]
    values = list(range(0, length - size + 1, stride))
    if values[-1] != length - size:
        values.append(length - size)
    return values


def main():
    target = OUT / "chip_geometry_feasibility.json"
    if target.exists():
        raise RuntimeError("geometry feasibility audit already frozen")
    eligible = [
        row for row in read_csv(OUT / "field_eligibility.csv")
        if row["split"] == "locked_external" and int(row["eligible_full_model"]) == 1
    ]
    maxima = []
    for item in eligible:
        field = RAW / item["country"] / item["field_id"]
        yield_path = field / "yield_masks/mean_scaled_yield_masked_regional_statistical_outlier.tif"
        values = tifffile.imread(yield_path)
        support = np.isfinite(values) & (values >= 0)
        h, w = support.shape
        best = 0.0
        best_pos = (0, 0)
        for y0 in positions(h):
            for x0 in positions(w):
                # The denominator remains 64x64; outside-grid pixels are padding.
                part = support[y0:min(y0 + 64, h), x0:min(x0 + 64, w)]
                coverage = float(part.sum() / (64 * 64))
                if coverage > best:
                    best, best_pos = coverage, (y0, x0)
        maxima.append({
            "field_id": item["field_id"], "country": item["country"],
            "best_coverage": best, "best_y0": best_pos[0], "best_x0": best_pos[1],
            "support_pixels": int(support.sum()), "height": h, "width": w,
        })
    levels = [0.70, 0.50, 0.30, 0.20, 0.10]
    report = {
        "scope": "locked_external_structurally_eligible_only",
        "outcome_blinding": "yield support mask only; no yield magnitudes aggregated or inspected",
        "window": 64,
        "placement_stride": 8,
        "fields": len(maxima),
        "field_count_by_best_coverage_threshold": {
            str(level): sum(row["best_coverage"] >= level for row in maxima)
            for level in levels
        },
        "best_coverage_quantiles": {
            str(q): float(np.quantile([row["best_coverage"] for row in maxima], q))
            for q in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
        },
        "fields_detail": maxima,
    }
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "fields_detail"}, indent=2))


if __name__ == "__main__":
    main()
