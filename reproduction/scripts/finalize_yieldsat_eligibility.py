"""Freeze structural exclusions after the read-only YieldSAT archive audit."""
from __future__ import annotations
import csv,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"results/protocol_v2_clean/accept_upgrade_v1/yieldsat_external"
def read(p):
    with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def main():
    target=OUT/"eligibility_report.json"
    if target.exists():raise RuntimeError("eligibility already frozen")
    audit=json.loads((OUT/"raw_structure_audit.json").read_text(encoding="utf-8"));bad={r["field_id"] for r in audit["errors"]};rows=read(OUT/"field_inventory.csv")
    out=[]
    for r in rows:
        reasons=[]
        if r["field_id"] in bad:reasons.append("yield_grid_not_exactly_aligned")
        if int(r["weather_complete"])!=1:reasons.append("missing_solar_radiation_for_frozen_full_model")
        out.append({**r,"eligible_full_model":int(not reasons),"exclusion_reasons":";".join(reasons)})
    with (OUT/"field_eligibility.csv").open("w",encoding="utf-8-sig",newline="") as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    eligible=[r for r in out if int(r["eligible_full_model"])==1];by_split=Counter(r["split"] for r in eligible);by_country=Counter(r["country"] for r in eligible)
    report={"status":"frozen_before_external_scoring","total_fields":len(out),"yield_grid_misaligned_excluded":len(bad),
            "missing_solar_radiation_excluded":sum(int(r["weather_complete"])!=1 for r in out),"eligible_full_model_fields":len(eligible),
            "eligible_by_split":dict(by_split),"eligible_by_country":dict(by_country),
            "locked_external_fields":by_split["locked_external"],"structural_rules":"no reprojection; no weather imputation"}
    target.write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
