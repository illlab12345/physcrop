"""Read-only structural audit of the downloaded YieldSAT flexible-format archive."""
from __future__ import annotations
import csv,hashlib,json,os,re
from collections import Counter
from pathlib import Path
import tifffile

ROOT=Path(__file__).resolve().parents[1];RAW=ROOT/"YieldSAT_raw_data";OUT=ROOT/"results/protocol_v2_clean/accept_upgrade_v1/yieldsat_external"
DATE=re.compile(r"(\d{8})")
def split(field_id):
    n=int(hashlib.sha256(("physcrop-yieldsat-v1:"+field_id).encode()).hexdigest()[:8],16)%100
    return "adapter_reference" if n<70 else ("audit_validation" if n<85 else "locked_external")
def shape(path):
    with tifffile.TiffFile(path) as t:return tuple(t.series[0].shape),str(t.series[0].dtype),t.pages[0].tags.get("ModelPixelScaleTag").value
def dates(paths):return [DATE.search(p.stem).group(1) for p in paths]
def main():
    OUT.mkdir(parents=True,exist_ok=True); target=OUT/"raw_structure_audit.json"
    if target.exists():raise RuntimeError("raw archive audit already frozen")
    records=[];errors=[];weather_schemas=Counter();countries=Counter();crops=Counter();years=Counter();splits=Counter()
    for country_dir in sorted(p for p in RAW.iterdir() if p.is_dir()):
      for field in sorted(p for p in country_dir.iterdir() if p.is_dir()):
        meta_files=list(field.glob("metadata-*.json")); weather_files=list((field/"weather").glob("*.csv"));
        s2=sorted((field/"s2_images").glob("S2_L2A_*.tif"));scl=sorted((field/"scl_masks").glob("S2_L2A_SCL_*.tif"));yield_path=field/"yield_masks/mean_scaled_yield_masked_regional_statistical_outlier.tif"
        local=[]
        if len(meta_files)!=1:local.append("metadata_count")
        if len(weather_files)!=1:local.append("weather_count")
        if not s2:local.append("no_s2")
        if dates(s2)!=dates(scl):local.append("s2_scl_date_mismatch")
        if not yield_path.exists():local.append("missing_yield")
        if local:errors.append({"field_id":field.name,"errors":local});continue
        meta=json.loads(meta_files[0].read_text(encoding="utf-8"));header=next(csv.reader(weather_files[0].open(encoding="utf-8-sig")))
        weather_schemas[",".join(header)]+=1; complete=all(x in header for x in ("Temp_mean","Temp_max","Temp_min","Total_prec","S_rad"))
        s2_shape,s2_dtype,pixel=shape(s2[0]);scl_shape,scl_dtype,scl_pixel=shape(scl[0]);y_shape,y_dtype,y_pixel=shape(yield_path)
        if len(s2_shape)!=3 or s2_shape[-1]!=12:local.append("s2_not_yx12")
        if s2_shape[:2]!=scl_shape or s2_shape[:2]!=y_shape:local.append("spatial_shape_mismatch")
        if pixel[:2]!=(10.0,10.0) or scl_pixel[:2]!=(10.0,10.0) or y_pixel[:2]!=(10.0,10.0):local.append("resolution_not_10m")
        if local:errors.append({"field_id":field.name,"errors":local})
        role=split(field.name);countries[country_dir.name]+=1;crops[meta["crop"]]+=1;years[int(meta["year"])]+=1;splits[role]+=1
        height,width=s2_shape[:2];padded_h=((height+63)//64)*64;padded_w=((width+63)//64)*64
        records.append({"field_id":field.name,"country":country_dir.name,"crop":meta["crop"],"year":meta["year"],"split":role,
                        "weather_complete":int(complete),"acquisitions":len(s2),"height":height,"width":width,
                        "padded_chip_grid":padded_h//64*padded_w//64,"s2_dtype":s2_dtype,"scl_dtype":scl_dtype,"yield_dtype":y_dtype,
                        "metadata_sha256":hashlib.sha256(meta_files[0].read_bytes()).hexdigest(),"weather_sha256":hashlib.sha256(weather_files[0].read_bytes()).hexdigest()})
    with (OUT/"field_inventory.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    report={"archive":"YieldSAT flexible format","fields":len(records),"errors":errors,"valid":not errors and len(records)==2173,
            "countries":dict(countries),"crops":dict(crops),"years":dict(sorted(years.items())),"field_splits":dict(splits),
            "weather_complete_fields":sum(r["weather_complete"] for r in records),"weather_incomplete_fields":sum(not r["weather_complete"] for r in records),
            "satellite_acquisitions":sum(r["acquisitions"] for r in records),"weather_schemas":dict(weather_schemas),
            "band_order_alias":["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"],
            "band_order_basis":"12-channel Sentinel-2 L2A stack omits B10; required channels map to indices 1,2,3,7,8,10",
            "units":{"reflectance":"DN scale 0..10000","temperature_input":"K","temperature_output":"C","precipitation_input":"m","precipitation_output":"mm"}}
    target.write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2))
    if not report["valid"]:raise SystemExit(2)
if __name__=="__main__":main()
