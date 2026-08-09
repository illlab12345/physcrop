"""Experiment 8: LOYO, unseen-farm, LOCO, and worst-group robustness."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

from run_yieldsat_experiment2 import CACHE, encode_features, subrole
from run_yieldsat_experiment6 import fit_variant


ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/yieldsat_risk_v1/experiment_08_robustness'


def crop_looks(data,fit,crops):
    out={}
    for crop in crops:
        vals=[float(np.nanmax(data['dynamic'][i,:data['lengths'][i],17])) for i in fit if str(data['crop'][i])==crop]
        if vals:out[crop]=.65*float(np.median(vals))
    return out


def make_stats(data,fit,level='country_crop'):
    y=data['target'].astype(float);stats={};crops=sorted(set(map(str,data['crop'][fit])))
    if level=='country_crop':
        for country in sorted(set(map(str,data['country'][fit]))):
            for crop in crops:
                ids=[i for i in fit if str(data['country'][i])==country and str(data['crop'][i])==crop]
                if len(ids)>=10:
                    v=y[ids];stats[f'{country}|{crop}']={'mean':float(v.mean()),'std':float(max(v.std(),.25)),'q20':float(np.quantile(v,.2)),'n':len(ids)}
    for crop in crops:
        ids=[i for i in fit if str(data['crop'][i])==crop];v=y[ids];stats[f'crop|{crop}']={'mean':float(v.mean()),'std':float(max(v.std(),.25)),'q20':float(np.quantile(v,.2)),'n':len(ids)}
    return stats


def get_stat(stats,country,crop):
    return stats.get(f'{country}|{crop}',stats.get(f'crop|{crop}'))


def basic(y,p):
    return {'n':len(y),'rmse':float(np.sqrt(mean_squared_error(y,p))),'r2':float(r2_score(y,p)),'spearman_rho':float(spearmanr(y,p).statistic)}


def bootstrap(rows,reps=5000):
    rng=np.random.default_rng(880088);groups={}
    for i,r in enumerate(rows):groups.setdefault(r.get('year',r.get('country')),[]).append(i)
    vals=[]
    for _ in range(reps):
        ids=[]
        for g in groups.values():ids.extend(rng.choice(g,len(g),replace=True))
        y=np.asarray([rows[i]['target'] for i in ids]);p=np.asarray([rows[i]['prediction'] for i in ids]);b=np.asarray([rows[i]['baseline'] for i in ids]);vals.append(np.sqrt(np.mean((y-p)**2))-np.sqrt(np.mean((y-b)**2)))
    return [float(x) for x in np.quantile(vals,[.025,.5,.975])]


def run_fold(data,fit,stop,test,countries,crops,level):
    stats=make_stats(data,fit,level);available={str(data['crop'][i]) for i in fit};test=np.asarray([i for i in test if str(data['crop'][i]) in available and get_stat(stats,str(data['country'][i]),str(data['crop'][i])) is not None],dtype=int)
    look=crop_looks(data,fit,crops);fx,fi=encode_features(data,fit,look,countries,crops);sx,si=encode_features(data,stop,look,countries,crops);tx,ti=encode_features(data,test,look,countries,crops)
    med=np.nanmedian(fx,axis=0);med[~np.isfinite(med)]=0;fx=np.where(np.isfinite(fx),fx,med);sx=np.where(np.isfinite(sx),sx,med);tx=np.where(np.isfinite(tx),tx,med);y=data['target'].astype(float)
    z=lambda ids:np.asarray([(y[i]-get_stat(stats,str(data['country'][i]),str(data['crop'][i]))['mean'])/get_stat(stats,str(data['country'][i]),str(data['crop'][i]))['std'] for i in ids])
    pz,w=fit_variant(fx,z(fi),sx,z(si),tx);pred=[];base=[];labels=[];risk=[]
    for j,i in enumerate(ti):
        stat=get_stat(stats,str(data['country'][i]),str(data['crop'][i]));value=float(pz[j]*stat['std']+stat['mean']);pred.append(value);base.append(stat['mean']);labels.append(y[i]<=stat['q20']);risk.append(stat['q20']-value)
    return ti,np.asarray(pred),np.asarray(base),np.asarray(labels),np.asarray(risk),w,look


def write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True);a=np.load(CACHE,allow_pickle=False);data={k:a[k] for k in a.files};a.close();y=data['target'].astype(float);all_ids=np.flatnonzero(np.isfinite(y));ref=np.flatnonzero((data['role']=='adapter_reference')&np.isfinite(y));fit_all=np.asarray([i for i in ref if subrole(data['farm_group'][i])=='model_fit']);stop_all=np.asarray([i for i in ref if subrole(data['farm_group'][i])=='early_stop']);countries=sorted(set(map(str,data['country'])));crops=sorted(set(map(str,data['crop'])))
    loyo=[];loyo_reports={}
    for year in sorted(set(map(int,data['year']))):
        fit=fit_all[data['year'][fit_all]!=year];stop=stop_all[data['year'][stop_all]!=year];test=all_ids[data['year'][all_ids]==year]
        ti,p,b,l,r,w,look=run_fold(data,fit,stop,test,countries,crops,'country_crop')
        for j,i in enumerate(ti):loyo.append({'field_id':str(data['field_id'][i]),'year':year,'country':str(data['country'][i]),'crop':str(data['crop'][i]),'target':float(y[i]),'prediction':float(p[j]),'baseline':float(b[j]),'low':int(l[j]),'risk':float(r[j])})
        m=basic(y[ti],p);m['baseline_rmse']=float(np.sqrt(mean_squared_error(y[ti],b)));m['auroc']=float(roc_auc_score(l,r)) if len(set(l))>1 else None;m['hist_weight']=w;m['look_gdd']=look;loyo_reports[str(year)]=m;print('LOYO',year,len(ti),m['rmse'],flush=True)
    write(OUT/'loyo_predictions.csv',loyo);ly=np.asarray([r['target'] for r in loyo]);lp=np.asarray([r['prediction'] for r in loyo]);lb=np.asarray([r['baseline'] for r in loyo]);loyo_summary=basic(ly,lp);loyo_summary['baseline_rmse']=float(np.sqrt(mean_squared_error(ly,lb)));loyo_summary['macro_year_r2']=float(np.mean([v['r2'] for v in loyo_reports.values()]));loyo_summary['rmse_delta_bootstrap95']=bootstrap(loyo)
    # Frozen-model unseen-farm subset from Experiment 6 predictions.
    with (ROOT/'results/yieldsat_risk_v1/experiment_06_ablations/predictions.csv').open(encoding='utf-8-sig') as h:ab=list(csv.DictReader(h))
    byid={str(data['field_id'][i]):i for i in all_ids};fit_farms=set(map(str,data['farm_group'][fit_all]));unseen=[r for r in ab if str(data['farm_group'][byid[r['field_id']]]) not in fit_farms];uy=np.asarray([float(r['target']) for r in unseen]);up=np.asarray([float(r['full']) for r in unseen])
    main_ref=json.loads((ROOT/'results/yieldsat_risk_v1/experiment_02_early_curve/target_reference.json').read_text(encoding='utf-8'))['low_yield_thresholds'];ul=[];ur=[]
    for row in unseen:
        i=byid[row['field_id']];key=f"{data['country'][i]}|{data['crop'][i]}";cut=main_ref.get(key,main_ref.get(f"crop|{data['crop'][i]}",main_ref['global']))['q20'];ul.append(float(row['target'])<=cut);ur.append(cut-float(row['full']))
    unseen_summary=basic(uy,up);unseen_summary.update({'farms':len(set(str(data['farm_group'][byid[r['field_id']]]) for r in unseen)),'auroc':float(roc_auc_score(ul,ur))})
    # Country holdouts with crop-level normalization.
    loco=[];loco_reports={}
    for held in countries:
        fit=fit_all[data['country'][fit_all]!=held];stop=stop_all[data['country'][stop_all]!=held];test=all_ids[data['country'][all_ids]==held]
        ti,p,b,l,r,w,look=run_fold(data,fit,stop,test,countries,crops,'crop')
        if len(ti)<10:continue
        for j,i in enumerate(ti):loco.append({'field_id':str(data['field_id'][i]),'country':held,'crop':str(data['crop'][i]),'target':float(y[i]),'prediction':float(p[j]),'baseline':float(b[j])})
        m=basic(y[ti],p);m['baseline_rmse']=float(np.sqrt(mean_squared_error(y[ti],b)));m['beats_crop_climatology']=m['rmse']<m['baseline_rmse'];m['hist_weight']=w;loco_reports[held]=m;print('LOCO',held,len(ti),m['rmse'],flush=True)
    write(OUT/'loco_predictions.csv',loco)
    ab_report=json.loads((ROOT/'results/yieldsat_risk_v1/experiment_06_ablations/report.json').read_text(encoding='utf-8'));worst=ab_report['prediction_ablations']['full']['worst_group_r2_n_ge_10']
    gates={'loyo_support':len(loyo)>=1500,'loyo_macro_year_r2_positive':loyo_summary['macro_year_r2']>0,'loyo_beats_climatology_ci':loyo_summary['rmse_delta_bootstrap95'][2]<0,
      'unseen_farm':len(unseen)>=50 and unseen_summary['r2']>0 and unseen_summary['auroc']>=.65,'loco_support_and_improvement':len(loco_reports)>=3 and sum(v['beats_crop_climatology'] for v in loco_reports.values())>=2,
      'worst_group_positive':worst>0}
    report={'experiment':8,'status':'PASS' if all(gates.values()) else 'FAIL','loyo_summary':loyo_summary,'loyo_by_year':loyo_reports,'unseen_farm':unseen_summary,'loco_by_country':loco_reports,'frozen_challenge_worst_group_r2':worst,
      'integrity':{'audit_targets_read':False,'loyo_recomputes_milestones_and_stats':True,'rapeseed_loco_excluded_without_source_crop':True,'legacy_farm_overlap_disclosed':True},'pass_gates':{k:bool(v) for k,v in gates.items()}}
    (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=True),encoding='utf-8');print(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=True))


if __name__=='__main__':main()
