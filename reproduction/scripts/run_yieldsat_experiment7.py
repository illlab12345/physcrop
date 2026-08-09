"""Experiment 7: field-clustered evaluation of auxiliary spatial yield maps."""
from __future__ import annotations

import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

from run_yieldsat_experiment2 import CACHE, OUT as EXP2_OUT, encode_features, subrole


ROOT=Path(__file__).resolve().parents[1];RAW=ROOT/'YieldSAT_raw_data';OUT=ROOT/'results/yieldsat_risk_v1/experiment_07_spatial_maps'


def chosen_date(data,i,look):
    n=int(data['lengths'][i]);gdd=data['dynamic'][i,:n,17];use=np.flatnonzero(np.isfinite(gdd)&(gdd<=look+1e-6))
    return date.fromordinal(int(data['dates'][i,use[-1]])).strftime('%Y%m%d') if len(use) else None


def raster2(path):
    a=np.asarray(tifffile.imread(path),dtype=np.float32)
    return np.nanmean(a,axis=2) if a.ndim==3 else a


def field_pixels(data,i,look,need_target=True):
    fid,country=str(data['field_id'][i]),str(data['country'][i]);folder=RAW/country/fid
    target=None
    if need_target:
        target=np.asarray(tifffile.imread(folder/'yield_masks/mean_scaled_yield_masked_regional_statistical_outlier.tif'),dtype=np.float32)
    n=int(data['lengths'][i]);gdd=data['dynamic'][i,:n,17];use=np.flatnonzero(np.isfinite(gdd)&(gdd<=look[str(data['crop'][i])]+1e-6))
    s2=scl=valid=None
    for pos in use[::-1]:
        text=date.fromordinal(int(data['dates'][i,pos])).strftime('%Y%m%d');s2p=folder/'s2_images'/f'S2_L2A_{text}.tif';sclp=folder/'scl_masks'/f'S2_L2A_SCL_{text}.tif'
        if not s2p.exists() or not sclp.exists():continue
        candidate_scl=np.asarray(tifffile.imread(sclp));candidate_valid=np.isin(candidate_scl,(4,5))
        if target is not None:candidate_valid&=np.isfinite(target)&(target>=0)
        if candidate_valid.sum()<50:continue
        candidate_s2=np.asarray(tifffile.imread(s2p),dtype=np.float32)/10000.
        if candidate_s2.ndim==3 and candidate_s2.shape[-1]>=12:
            s2,scl,valid=candidate_s2,candidate_scl,candidate_valid;break
    if s2 is None:return None
    b02,b04,b05,b07,b08,b11=s2[:,:,1],s2[:,:,3],s2[:,:,4],s2[:,:,6],s2[:,:,7],s2[:,:,10]
    def ratio(num,den):
        out=np.zeros_like(num,dtype=np.float32);np.divide(num,den,out=out,where=np.abs(den)>1e-4);return out
    extra=[ratio(b08-b04,b08+b04),ratio(b08-b11,b08+b11),ratio(b07-b05,b07+b05),ratio(2.5*(b08-b04),b08+6*b04-7.5*b02+1)]
    layers=[s2[:,:,j] for j in range(12)]+extra
    for stem in ('cec','cfvo','clay','nitrogen','phh2o','sand','silt','soc'):
        p=next((folder/'soil').glob(f'{stem}_*.tif'),None);layers.append(raster2(p) if p else np.full(scl.shape,np.nan))
    for stem in ('aspect','curvature','dem','slope','twi'):
        p=next((folder/'dem').glob(f'{stem}-*.tif'),None);layers.append(raster2(p) if p else np.full(scl.shape,np.nan))
    x=np.stack(layers,axis=2)
    if valid.sum()<50:return None
    for col in range(x.shape[2]):
        layer=x[:,:,col];values=layer[valid];finite=np.isfinite(values);fill=float(np.median(values[finite])) if finite.any() else 0.0
        layer[~np.isfinite(layer)]=fill;x[:,:,col]=layer
    vals=x[valid];mean=vals.mean(0);std=vals.std(0);std[std<1e-6]=1.;vals=(vals-mean)/std
    return {'field_id':fid,'features':vals.astype(np.float32),'target':target[valid].astype(np.float32) if target is not None else None,'mask':valid,'shape':valid.shape,'ndvi_col':12}


def training_one(data,i,look):
    item=field_pixels(data,i,look,True)
    if item is None:return None
    y=item['target'];res=y-y.mean();n=min(256,len(y));seed=int(hashlib.sha256(item['field_id'].encode()).hexdigest()[:8],16);sel=np.random.default_rng(seed).choice(len(y),n,replace=False)
    return item['features'][sel],res[sel]


def bootstrap_delta(rows,a,b,reps=5000):
    rng=np.random.default_rng(77007);vals=[]
    for _ in range(reps):
        ids=rng.integers(0,len(rows),len(rows));vals.append(np.mean([rows[i][a]-rows[i][b] for i in ids]))
    return [float(x) for x in np.quantile(vals,[.025,.5,.975])]


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'maps').mkdir(exist_ok=True)
    canonical=OUT/'report.json'
    if canonical.exists() and not (OUT/'report_attempt3.json').exists():
        (OUT/'report_attempt3.json').write_text(canonical.read_text(encoding='utf-8'),encoding='utf-8')
    archive=np.load(CACHE,allow_pickle=False)
    # NpzFile performs lazy decompression and is not safe for concurrent reads.
    data={key:archive[key] for key in archive.files};archive.close()
    y=data['target'].astype(float)
    ref=np.flatnonzero((data['role']=='adapter_reference')&np.isfinite(y));fit=np.asarray([i for i in ref if subrole(data['farm_group'][i])=='model_fit']);test=np.flatnonzero((data['role']=='locked_external')&np.isfinite(y));bundle=joblib.load(EXP2_OUT/'model_look3.joblib');look=bundle['look_gdd']
    chunks=[];errors=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(training_one,data,int(i),look):int(i) for i in fit}
        for done,f in enumerate(as_completed(futures),1):
            try:
                value=f.result();chunks.append(value) if value is not None else errors.append(str(data['field_id'][futures[f]]))
            except Exception as exc:errors.append(f"{data['field_id'][futures[f]]}:{exc}")
            if done==1 or done%100==0:print(f'train_fields={done}/{len(fit)} usable={len(chunks)}',flush=True)
    train_x=np.concatenate([x for x,_ in chunks]);train_r=np.concatenate([r for _,r in chunks]);model=HistGradientBoostingRegressor(max_iter=300,min_samples_leaf=50,l2_regularization=1.0,random_state=7701);model.fit(train_x,train_r);ridge=Ridge(alpha=10.0).fit(train_x[:,[12]],train_r)
    # Frozen field-level means for challenge fields.
    tx,ti=encode_features(data,test,look,bundle['countries'],bundle['crops']);tx=np.where(np.isfinite(tx),tx,bundle['feature_medians']);z=(1-bundle['hist_weight'])*bundle['model'].predict(tx)+bundle['hist_weight']*bundle['hist_model'].predict(tx);field_pred={str(data['field_id'][i]):float(z[j]*bundle['target_stats'][f"{data['country'][i]}|{data['crop'][i]}"]['std']+bundle['target_stats'][f"{data['country'][i]}|{data['crop'][i]}"]['mean']) for j,i in enumerate(ti)}
    selected=set(sorted(field_pred,key=lambda x:hashlib.sha256(('spatial-map:'+x).encode()).hexdigest())[:8]);rows=[];maps={};eval_errors=[];max_preserve=0.;total_pixels=0
    for done,i in enumerate(test,1):
        item=field_pixels(data,int(i),look,True)
        if item is None or item['field_id'] not in field_pred:eval_errors.append(str(data['field_id'][i]));continue
        x,truth=item['features'],item['target'];base=field_pred[item['field_id']];res=model.predict(x);res-=res.mean();nd= ridge.predict(x[:,[12]]);nd-=nd.mean();pred=base+res;ndpred=base+nd;uniform=np.full(len(truth),base);max_preserve=max(max_preserve,abs(pred.mean()-base));total_pixels+=len(truth)
        q=np.quantile(truth,.2);label=truth<=q
        rho=spearmanr(truth,pred).statistic;ndrho=spearmanr(truth,ndpred).statistic
        row={'field_id':item['field_id'],'country':str(data['country'][i]),'crop':str(data['crop'][i]),'pixels':len(truth),'field_prediction':base,
             'rmse_spatial':float(np.sqrt(mean_squared_error(truth,pred))),'rmse_ndvi':float(np.sqrt(mean_squared_error(truth,ndpred))),'rmse_uniform':float(np.sqrt(mean_squared_error(truth,uniform))),
             'r2_spatial':float(r2_score(truth,pred)),'rho_spatial':float(rho),'rho_ndvi':float(ndrho),'bottom20_auroc':float(roc_auc_score(label,-pred))}
        rows.append(row)
        if item['field_id'] in selected:
            arrays={};
            for name,values in [('target',truth),('prediction',pred),('ndvi',ndpred)]:
                a=np.full(item['shape'],np.nan,dtype=np.float32);a[item['mask']]=values;arrays[name]=a
            np.savez_compressed(OUT/'maps'/f"{item['field_id']}.npz",**arrays);maps[item['field_id']]=arrays
        if done==1 or done%50==0:print(f'eval_fields={done}/{len(test)} usable={len(rows)}',flush=True)
    joblib.dump({'model':model,'ndvi_ridge':ridge},OUT/'spatial_residual_models.joblib')
    with (OUT/'field_metrics.csv').open('w',encoding='utf-8-sig',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Fixed, outcome-blind qualitative IDs; display first four for compactness.
    fig,axes=plt.subplots(min(4,len(maps)),3,figsize=(9,3*min(4,len(maps))),squeeze=False)
    for r,(fid,a) in enumerate(list(maps.items())[:4]):
        vmin=np.nanpercentile(a['target'],5);vmax=np.nanpercentile(a['target'],95)
        for c,name in enumerate(('target','prediction','ndvi')):axes[r,c].imshow(a[name],vmin=vmin,vmax=vmax,cmap='viridis');axes[r,c].set_title(f'{fid[:24]}\n{name}');axes[r,c].axis('off')
    fig.tight_layout();fig.savefig(OUT/'qualitative_maps.png',dpi=180);plt.close(fig)
    ci_uniform=bootstrap_delta(rows,'rmse_spatial','rmse_uniform');ci_ndvi=bootstrap_delta(rows,'rmse_spatial','rmse_ndvi');median_rho=float(np.median([r['rho_spatial'] for r in rows]));median_auc=float(np.median([r['bottom20_auroc'] for r in rows]))
    summary={'fields':len(rows),'pixels':total_pixels,'training_fields':len(chunks),'training_pixels':len(train_r),'mean_field_rmse_spatial':float(np.mean([r['rmse_spatial'] for r in rows])),
      'mean_field_rmse_uniform':float(np.mean([r['rmse_uniform'] for r in rows])),'mean_field_rmse_ndvi':float(np.mean([r['rmse_ndvi'] for r in rows])),
      'median_within_field_rho':median_rho,'median_bottom20_auroc':median_auc,'spatial_minus_uniform_bootstrap95':ci_uniform,'spatial_minus_ndvi_bootstrap95':ci_ndvi,'max_field_mean_preservation_error':max_preserve}
    gates={'support':bool(len(rows)>=250 and total_pixels>=100000),'mean_preservation':bool(max_preserve<=1e-5),'beats_uniform_ci':bool(ci_uniform[2]<0),'median_rho_ge_0.20':bool(median_rho>=.2),'median_bottom20_auroc_ge_0.65':bool(median_auc>=.65),'field_clustered_inference_only':True}
    report={'experiment':7,'attempt':4,'status':'PASS' if all(gates.values()) else 'FAIL','summary':summary,'qualitative_ids':sorted(maps),'excluded_training_fields':len(errors),'excluded_challenge_fields':len(eval_errors),'integrity':{'audit_targets_read':False,'future_images_used':False,'pixel_iid_inference':False},'pass_gates':gates}
    (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=True),encoding='utf-8');print(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=True))


if __name__=='__main__':main()
