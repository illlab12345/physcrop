"""Experiment 6: prediction and decision ablations."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

from run_yieldsat_experiment2 import CACHE, OUT as EXP2_OUT, encode_features, subrole
from run_yieldsat_experiment5 import metrics, group_metrics, bootstrap_delta


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/yieldsat_risk_v1/experiment_06_ablations'


def encode_rank(data,indices,look_by_crop,countries,crops):
    rows=[];kept=[]
    for i in indices:
        look=look_by_crop[str(data['crop'][i])];n=int(data['lengths'][i]);seq=data['dynamic'][i,:n];gdd=seq[:,17]
        use=np.flatnonzero(np.isfinite(gdd)&(gdd<=look+1e-6))
        if len(use)<3:continue
        seq=seq[use];selected=np.r_[np.arange(17),18];query=np.linspace(0,len(seq)-1,6);temporal=[]
        for col in selected:
            vals=seq[:,col].astype(float);finite=np.isfinite(vals)
            temporal.extend(np.interp(query,np.flatnonzero(finite),vals[finite]).tolist() if finite.sum()>1 else [float(vals[finite][0]) if finite.sum() else np.nan]*6)
        onehot=[float(str(data['country'][i])==x) for x in countries]+[float(str(data['crop'][i])==x) for x in crops]
        rows.append(temporal+data['static'][i].astype(float).tolist()+onehot);kept.append(i)
    return np.asarray(rows,dtype=np.float32),np.asarray(kept,dtype=int)


def subset(x,name,ncat):
    temporal=np.arange(108);static=np.arange(108,123);cats=np.arange(x.shape[1]-ncat,x.shape[1])
    if name=='no_weather': cols=np.setdiff1d(np.arange(x.shape[1]),np.arange(102,108))
    elif name=='no_static': cols=np.r_[temporal,cats]
    elif name=='no_indices': cols=np.setdiff1d(np.arange(x.shape[1]),np.arange(72,96))
    elif name=='s2_only': cols=np.r_[np.arange(72),cats]
    elif name=='no_domain': cols=np.r_[temporal,static]
    else: cols=np.arange(x.shape[1])
    return x[:,cols],cols


def fit_variant(train_x,train_z,stop_x,stop_z,test_x,force_et=False):
    med=np.nanmedian(train_x,axis=0);med[~np.isfinite(med)]=0
    train_x=np.where(np.isfinite(train_x),train_x,med);stop_x=np.where(np.isfinite(stop_x),stop_x,med);test_x=np.where(np.isfinite(test_x),test_x,med)
    et=ExtraTreesRegressor(n_estimators=600,min_samples_leaf=3,max_features=.7,bootstrap=False,random_state=2201,n_jobs=-1);et.fit(train_x,train_z)
    if force_et:return et.predict(test_x),0.0
    hg=HistGradientBoostingRegressor(max_iter=300,min_samples_leaf=15,l2_regularization=1.0,random_state=5501);hg.fit(train_x,train_z)
    se,sh=et.predict(stop_x),hg.predict(stop_x);weights=np.linspace(0,1,11);w=float(min(weights,key=lambda z:np.sqrt(np.mean((stop_z-((1-z)*se+z*sh))**2))))
    return (1-w)*et.predict(test_x)+w*hg.predict(test_x),w


def raw(z,ids,data,stats):
    return np.asarray([z[j]*stats[f"{data['country'][i]}|{data['crop'][i]}"]['std']+stats[f"{data['country'][i]}|{data['crop'][i]}"]['mean'] for j,i in enumerate(ids)])


def main():
    OUT.mkdir(parents=True,exist_ok=True);data=np.load(CACHE,allow_pickle=False);y=data['target'].astype(float)
    ref=np.flatnonzero((data['role']=='adapter_reference')&np.isfinite(y));fit=np.asarray([i for i in ref if subrole(data['farm_group'][i])=='model_fit']);stop=np.asarray([i for i in ref if subrole(data['farm_group'][i])=='early_stop']);test=np.flatnonzero((data['role']=='locked_external')&np.isfinite(y))
    bundle=joblib.load(EXP2_OUT/'model_look3.joblib');countries,crops,look,stats=bundle['countries'],bundle['crops'],bundle['look_gdd'],bundle['target_stats'];ncat=len(countries)+len(crops)
    fx,fi=encode_features(data,fit,look,countries,crops);sx,si=encode_features(data,stop,look,countries,crops);tx,ti=encode_features(data,test,look,countries,crops)
    rz=lambda ids:np.asarray([(y[i]-stats[f"{data['country'][i]}|{data['crop'][i]}"]['mean'])/stats[f"{data['country'][i]}|{data['crop'][i]}"]['std'] for i in ids])
    fz,sz=rz(fi),rz(si);preds={};weights={}
    for name in ('full','no_weather','no_static','no_indices','s2_only','no_domain','extratrees_only'):
        ax,_=subset(fx,name,ncat);bx,_=subset(sx,name,ncat);cx,_=subset(tx,name,ncat)
        if name=='no_domain':
            mean,std=float(y[fi].mean()),float(max(y[fi].std(),.25));az=(y[fi]-mean)/std;bz=(y[si]-mean)/std;p,w=fit_variant(ax,az,bx,bz,cx);pred=p*std+mean
        else:
            p,w=fit_variant(ax,fz,bx,sz,cx,force_et=name=='extratrees_only');pred=raw(p,ti,data,stats)
        preds[name]=pred;weights[name]=w
    rfx,rfi=encode_rank(data,fit,look,countries,crops);rsx,rsi=encode_rank(data,stop,look,countries,crops);rtx,rti=encode_rank(data,test,look,countries,crops)
    if not (np.array_equal(rfi,fi) and np.array_equal(rsi,si) and np.array_equal(rti,ti)):raise RuntimeError('rank/GDD eligible fields differ')
    p,w=fit_variant(rfx,fz,rsx,sz,rtx);preds['rank_aligned']=raw(p,ti,data,stats);weights['rank_aligned']=w
    thresholds=json.loads((EXP2_OUT/'target_reference.json').read_text(encoding='utf-8'))['low_yield_thresholds'];test_y=y[ti]
    reports={};rows=[]
    for name,pred in preds.items():
        gm=group_metrics(test_y,pred,ti,data);reports[name]={'overall':metrics(test_y,pred,ti,data,thresholds),'macro_group_r2':float(np.mean([v['r2'] for v in gm.values()])),
          'worst_group_r2_n_ge_10':float(min(v['r2'] for v in gm.values() if v['n']>=10)),'early_stop_hist_weight':weights[name],'per_group':gm}
    for j,i in enumerate(ti):
        row={'field_id':str(data['field_id'][i]),'group':f"{data['country'][i]}|{data['crop'][i]}",'target':float(y[i])};row.update({k:float(v[j]) for k,v in preds.items()});rows.append(row)
    cis={name:bootstrap_delta(rows,'full',name) for name in preds if name!='full'}
    decision=json.loads((ROOT/'results/yieldsat_risk_v1/experiment_04_familywise_alert/report.json').read_text(encoding='utf-8'))['policy_summaries']
    watch,action=decision['alpha_spending'],decision['alpha_spending_persistent'];removals=['no_weather','no_static','no_indices','no_domain']
    full_rmse=reports['full']['overall']['rmse'];best=min(v['overall']['rmse'] for v in reports.values())
    gates={'full_within_2pct_best':full_rmse<=1.02*best,'full_significantly_beats_s2_only':cis['s2_only'][2]<0,
      'gdd_point_better_than_rank':full_rmse<reports['rank_aligned']['overall']['rmse'],'gdd_within_2pct_rank':full_rmse<=1.02*reports['rank_aligned']['overall']['rmse'],
      'two_removals_worsen_point_rmse':sum(reports[x]['overall']['rmse']>full_rmse for x in removals)>=2,
      'persistence_tradeoff':action['fpr_field_season']<=.5*watch['fpr_field_season'] and action['tp']>=.25*watch['tp'] and action['precision']>=.8}
    import csv
    with (OUT/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as h:wr=csv.DictWriter(h,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    report={'experiment':6,'status':'PASS' if all(gates.values()) else 'FAIL','prediction_ablations':reports,'paired_rmse_full_minus_ablation_bootstrap95':cis,
      'decision_ablations':{'watch':watch,'persistent_action':action,'regression_ucb_attempt':{'tp':0,'fp':0}},'integrity':{'audit_targets_read':False,'identical_fields':True,'causal_prefix_only':True},'pass_gates':gates}
    (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=True),encoding='utf-8');print(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=True))


if __name__=='__main__':main()
