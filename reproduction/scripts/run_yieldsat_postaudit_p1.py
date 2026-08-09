"""P1 timing-alignment ladder on the development challenge only."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_squared_error, r2_score, roc_auc_score

from run_yieldsat_experiment2 import INTERP_FRACTIONS, encode_features, subrole, threshold_for


ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/yieldsat_risk_v1';EXP2=BASE/'experiment_02_early_curve';OUT=BASE/'post_audit_p0p1'
SELECTED=np.r_[np.arange(17),18]


def temporal_row(seq,x,query):
    temporal=[]
    for col in SELECTED:
        v=seq[:,col].astype(float);finite=np.isfinite(v)&np.isfinite(x)
        if finite.sum()==0: temporal.extend([np.nan]*len(query))
        elif finite.sum()==1: temporal.extend([float(v[finite][0])]*len(query))
        else: temporal.extend(np.interp(query,x[finite],v[finite]).tolist())
    return temporal


def prefix(data,i,look):
    n=int(data['lengths'][i]);seq=data['dynamic'][i,:n];gdd=seq[:,17].astype(float);use=np.flatnonzero(np.isfinite(gdd)&(gdd<=look+1e-6))
    return (seq[use],data['dates'][i,use].astype(float),gdd[use]) if len(use)>=3 else (None,None,None)


def calendar_references(data,fit,look_by_crop):
    spans={}
    for i in fit:
        crop=str(data['crop'][i]);seq,days,gdd=prefix(data,i,look_by_crop[crop])
        if seq is not None:spans.setdefault(f"{data['country'][i]}|{crop}",[]).append(float(days[-1]-days[0]))
    return {k:max(1.0,float(np.median(v))) for k,v in spans.items()}


def encode(data,indices,look_by_crop,countries,crops,method,calendar_span):
    if method=='gdd': return encode_features(data,indices,look_by_crop,countries,crops)
    rows=[];kept=[]
    for i in indices:
        crop=str(data['crop'][i]);seq,days,gdd=prefix(data,i,look_by_crop[crop])
        if seq is None:continue
        if method=='acquisition_rank':x=np.arange(len(seq),dtype=float);query=np.linspace(0,len(seq)-1,6)
        elif method=='normalized_season_fraction':
            x=(days-days[0])/max(1.0,days[-1]-days[0]);query=INTERP_FRACTIONS
        elif method=='calendar_doy':
            # Calendar time since the first in-season usable acquisition; the
            # country-crop reference span is learned from model-fit fields only.
            x=days-days[0];query=INTERP_FRACTIONS*calendar_span[f"{data['country'][i]}|{crop}"]
        else:raise ValueError(method)
        temporal=temporal_row(seq,x,query);one=[float(str(data['country'][i])==c) for c in countries]+[float(crop==c) for c in crops]
        rows.append(temporal+data['static'][i].astype(float).tolist()+one);kept.append(int(i))
    return np.asarray(rows,dtype=np.float32),np.asarray(kept,dtype=int)


def raw(z,ids,data,stats):
    return np.asarray([z[j]*stats[f"{data['country'][i]}|{data['crop'][i]}"]['std']+stats[f"{data['country'][i]}|{data['crop'][i]}"]['mean'] for j,i in enumerate(ids)])


def fit_predict(fx,fi,sx,si,tx,ti,data,stats):
    y=data['target'].astype(float);med=np.nanmedian(fx,axis=0);med[~np.isfinite(med)]=0;fx=np.where(np.isfinite(fx),fx,med);sx=np.where(np.isfinite(sx),sx,med);tx=np.where(np.isfinite(tx),tx,med)
    z=lambda ids:np.asarray([(y[i]-stats[f"{data['country'][i]}|{data['crop'][i]}"]['mean'])/stats[f"{data['country'][i]}|{data['crop'][i]}"]['std'] for i in ids]);fz,sz=z(fi),z(si)
    et=ExtraTreesRegressor(n_estimators=600,min_samples_leaf=3,max_features=.7,bootstrap=False,random_state=2201,n_jobs=-1).fit(fx,fz);hg=HistGradientBoostingRegressor(max_iter=300,min_samples_leaf=15,l2_regularization=1.0,random_state=5501).fit(fx,fz)
    ep,hp=et.predict(sx),hg.predict(sx);weights=np.linspace(0,1,11);w=float(min(weights,key=lambda q:np.sqrt(np.mean((sz-((1-q)*ep+q*hp))**2))))
    return raw((1-w)*et.predict(tx)+w*hg.predict(tx),ti,data,stats),w


def summarize(y,p,b,ids,data,thresholds):
    cut=np.asarray([threshold_for(thresholds,str(data['country'][i]),str(data['crop'][i]))[0] for i in ids]);lab=y<=cut;rmse=float(np.sqrt(mean_squared_error(y,p)));br=float(np.sqrt(mean_squared_error(y,b)));gm={}
    for group in sorted(set(f"{data['country'][i]}|{data['crop'][i]}" for i in ids)):
        pos=np.asarray([j for j,i in enumerate(ids) if f"{data['country'][i]}|{data['crop'][i]}"==group]);gm[group]={'n':len(pos),'rmse':float(np.sqrt(mean_squared_error(y[pos],p[pos]))),'r2':float(r2_score(y[pos],p[pos]))}
    return {'n':len(y),'rmse':rmse,'r2':float(r2_score(y,p)),'nrmse_by_target_sd':float(rmse/np.std(y)),'climatology_rmse':br,'rmse_skill':float(1-rmse/br),'mse_skill':float(1-rmse**2/br**2),'prevalence':float(np.mean(lab)),'auroc':float(roc_auc_score(lab,cut-p)),'auprc':float(average_precision_score(lab,cut-p)),'macro_group_r2':float(np.mean([v['r2'] for v in gm.values()])),'worst_group_r2':float(min(v['r2'] for v in gm.values())),'per_group':gm}


def paired_bootstrap(rows,a,b,reps=5000):
    groups={};
    for j,r in enumerate(rows):groups.setdefault(r['group'],[]).append(j)
    rng=np.random.default_rng(260801);vals=[]
    for _ in range(reps):
        ids=[]
        for g in groups.values():ids.extend(rng.choice(g,len(g),replace=True))
        y=np.asarray([rows[j]['target'] for j in ids]);pa=np.asarray([rows[j][a] for j in ids]);pb=np.asarray([rows[j][b] for j in ids]);vals.append(np.sqrt(np.mean((y-pa)**2))-np.sqrt(np.mean((y-pb)**2)))
    return [float(x) for x in np.quantile(vals,[.025,.5,.975])]


def main():
    OUT.mkdir(parents=True,exist_ok=True);a=np.load(BASE/'cache/development_sequences.npz',allow_pickle=False);data={k:a[k] for k in a.files};a.close();y=data['target'].astype(float)
    ref=np.flatnonzero((data['role']=='adapter_reference')&np.isfinite(y));fit=np.asarray([i for i in ref if subrole(data['farm_group'][i])=='model_fit']);stop=np.asarray([i for i in ref if subrole(data['farm_group'][i])=='early_stop']);test=np.flatnonzero((data['role']=='locked_external')&np.isfinite(y))
    bundle=joblib.load(EXP2/'model_look3.joblib');countries,crops,look,stats=bundle['countries'],bundle['crops'],bundle['look_gdd'],bundle['target_stats'];thresholds=json.loads((EXP2/'target_reference.json').read_text(encoding='utf-8'))['low_yield_thresholds'];span=calendar_references(data,fit,look)
    methods=('calendar_doy','acquisition_rank','normalized_season_fraction','gdd');preds={};weights={};eligible={}
    for method in methods:
        fx,fi=encode(data,fit,look,countries,crops,method,span);sx,si=encode(data,stop,look,countries,crops,method,span);tx,ti=encode(data,test,look,countries,crops,method,span);eligible[method]=(fi,si,ti)
        preds[method],weights[method]=fit_predict(fx,fi,sx,si,tx,ti,data,stats)
    first=eligible[methods[0]]
    if not all(all(np.array_equal(a,b) for a,b in zip(first,eligible[m])) for m in methods):raise RuntimeError('Alignment ladder uses different fields')
    ti=first[2];yy=y[ti];bb=np.asarray([stats[f"{data['country'][i]}|{data['crop'][i]}"]['mean'] for i in ti]);reports={m:summarize(yy,preds[m],bb,ti,data,thresholds) for m in methods};rows=[]
    for j,i in enumerate(ti):
        r={'field_id':str(data['field_id'][i]),'group':f"{data['country'][i]}|{data['crop'][i]}",'target':float(yy[j])};r.update({m:float(preds[m][j]) for m in methods});rows.append(r)
    cis={m:paired_bootstrap(rows,'gdd',m) for m in methods if m!='gdd'}
    with (OUT/'p1_alignment_predictions.csv').open('w',encoding='utf-8-sig',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'analysis_status':'post-audit P1 development-only analysis; audit outcomes unused','fixed_prefix':'crop-specific 65% absolute-GDD look; alignment coordinate alone varies','calendar_definition':'elapsed calendar days from first in-season usable acquisition, with country-crop reference span estimated on model-fit fields','methods':reports,'early_stop_hist_weights':weights,'gdd_minus_alternative_rmse_bootstrap95':cis,'calendar_reference_span_days':span,'integrity':{'audit_outcomes_read':False,'identical_model_family':True,'identical_feature_budget':True,'identical_fields':True,'challenge_fields':len(ti),'model_fit_fields':len(first[0]),'early_stop_fields':len(first[1])}}
    (OUT/'p1_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
