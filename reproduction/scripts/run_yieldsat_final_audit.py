"""Preflight or execute the unique frozen YieldSAT audit outcome reveal."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np,tifffile
from scipy.stats import beta,spearmanr
from sklearn.metrics import average_precision_score,mean_absolute_error,mean_squared_error,r2_score,roc_auc_score

from run_yieldsat_experiment2 import subrole,threshold_for
from run_yieldsat_experiment3 import predictions,build_cells,choose_cell,conformal_quantile

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/yieldsat_risk_v1';OUT=BASE/'final_audit';MANIFEST=OUT/'freeze_manifest.json';CACHE=BASE/'cache/audit_features_blind.npz';DEV=BASE/'cache/development_sequences.npz';RAW=ROOT/'YieldSAT_raw_data'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def verify():
 m=json.loads(MANIFEST.read_text(encoding='utf-8'));bad=[]
 for rel,want in m['hashes'].items():
  p=ROOT/rel;got=sha(p) if p.exists() else 'MISSING'
  if got!=want:bad.append({'file':rel,'expected':want,'actual':got})
 if bad:raise RuntimeError(f'Frozen hash mismatch: {bad}')
 a=np.load(CACHE,allow_pickle=False)
 if np.isfinite(a['target']).any() or len(a['field_id'])!=m['audit_fields']:raise RuntimeError('Blind cache integrity failed')
 return m

def reveal_targets(data):
 out=[]
 for fid,country in zip(data['field_id'],data['country']):
  path=RAW/str(country)/str(fid)/'yield_masks/mean_scaled_yield_masked_regional_statistical_outlier.tif';v=np.asarray(tifffile.imread(path),dtype=np.float32);keep=np.isfinite(v)&(v>=0)
  out.append(float(v[keep].mean()) if keep.any() else np.nan)
 return np.asarray(out)

def metric(y,p,labels,risk):
 rho=spearmanr(y,p).statistic
 return {'n':int(len(y)),'rmse':float(np.sqrt(mean_squared_error(y,p))),'mae':float(mean_absolute_error(y,p)),'r2':float(r2_score(y,p)),'spearman_rho':float(rho),
  'prevalence':float(np.mean(labels)),'auroc':float(roc_auc_score(labels,risk)),'auprc':float(average_precision_score(labels,risk))}

def bootstrap(rows,reps=5000):
 rng=np.random.default_rng(999001);groups={}
 for i,r in enumerate(rows):groups.setdefault(r['group'],[]).append(i)
 vals=[]
 for _ in range(reps):
  ids=[]
  for g in groups.values():ids.extend(rng.choice(g,len(g),replace=True))
  y=np.asarray([rows[i]['target'] for i in ids]);p=np.asarray([rows[i]['prediction'] for i in ids]);b=np.asarray([rows[i]['baseline'] for i in ids]);vals.append(np.sqrt(np.mean((y-p)**2))-np.sqrt(np.mean((y-b)**2)))
 return [float(x) for x in np.quantile(vals,[.025,.5,.975])]

def upper_cp(x,n):return 1.0 if x==n else float(beta.ppf(.95,x+1,n-x))

def args():
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--preflight',action='store_true');g.add_argument('--execute-once',action='store_true');return p.parse_args()

def main():
 a=args();m=verify();result_path=OUT/'final_report.json';marker=OUT/'REVEAL_STARTED.json'
 if result_path.exists():raise SystemExit('Final audit already completed; refusing repetition')
 if a.preflight:
  if marker.exists():raise RuntimeError('Reveal marker already exists')
  print(json.dumps({'status':'PREFLIGHT_PASS','audit_fields':m['audit_fields'],'hashes_verified':len(m['hashes']),'targets_read':False},indent=2));return
 if marker.exists():raise RuntimeError('A prior reveal started; refusing repetition')
 marker.write_text(json.dumps({'started_utc':datetime.now(timezone.utc).isoformat(),'freeze_manifest_sha256':sha(MANIFEST)},indent=2),encoding='utf-8')
 archive=np.load(CACHE,allow_pickle=False);audit={k:archive[k] for k in archive.files};archive.close();audit['target']=reveal_targets(audit)
 devarc=np.load(DEV,allow_pickle=False);dev={k:devarc[k] for k in devarc.files};devarc.close();dy=dev['target'].astype(float)
 thresholds=json.loads((BASE/'experiment_02_early_curve/target_reference.json').read_text(encoding='utf-8'))['low_yield_thresholds'];audit_ids=np.arange(len(audit['field_id']));ref=np.flatnonzero((dev['role']=='adapter_reference')&np.isfinite(dy));cal=np.asarray([i for i in ref if subrole(dev['farm_group'][i])=='conformal']);nullcal=np.asarray([i for i in ref if subrole(dev['farm_group'][i]) in ('early_stop','conformal')])
 selected={1:'global_raw',2:'hierarchy_power50',3:'global_raw'};alphas={1:.010,2:.015,3:.025};flat=[];looks={};watch_by={str(x):{} for x in audit['field_id']};truth={};groups={}
 for i in audit_ids:
  fid=str(audit['field_id'][i]);cut,_=threshold_for(thresholds,str(audit['country'][i]),str(audit['crop'][i]));truth[fid]=int(audit['target'][i]<=cut);groups[fid]=f"{audit['country'][i]}|{audit['crop'][i]}"
 for look in (1,2,3):
  bundle=joblib.load(BASE/f'experiment_02_early_curve/model_look{look}.joblib');evalrows=predictions(bundle,audit,audit_ids);calrows=predictions(bundle,dev,cal);nullrows=predictions(bundle,dev,nullcal)
  for r in calrows:r['raw_score']=r['target']-r['prediction'];r['power50_score']=r['raw_score']/np.sqrt(r['scale'])
  rawcells=build_cells(calrows,'raw_score');powercells=build_cells(calrows,'power50_score');method=selected[look];cells=powercells if method=='hierarchy_power50' else rawcells
  qsets={nom:{k:conformal_quantile(v,nom) for k,v in cells.items()} for nom in (.90,.95)}
  nullscores=[]
  for r in nullrows:
   cut,_=threshold_for(thresholds,r['country'],r['crop']);stat=bundle['target_stats'][r['group']]
   if r['target']>cut:nullscores.append((cut-r['prediction'])/stat['std'])
  nullscores=np.asarray(nullscores);rows=[]
  for r in evalrows:
   fid=r['field_id'];cut,_=threshold_for(thresholds,r['country'],r['crop']);stat=bundle['target_stats'][r['group']];cell=choose_cell(method,r,cells);entry={'look':look,**r,'baseline':stat['mean'],'cutoff':cut,'true_low':truth[fid],'method':method}
   for nom in (.90,.95):
    q,_rank=qsets[nom][cell];adj=q*np.sqrt(r['scale']) if method=='hierarchy_power50' else q;upper=r['prediction']+adj;entry[f'upper_{int(nom*100)}']=float(upper);entry[f'covered_{int(nom*100)}']=int(r['target']<=upper)
   score=(cut-r['prediction'])/stat['std'];pvalue=float((1+np.sum(nullscores>=score))/(len(nullscores)+1));watch=int(pvalue<=alphas[look]);entry.update(score=float(score),pvalue=pvalue,alpha=alphas[look],watch=watch);watch_by[fid][look]=watch;rows.append(entry);flat.append(entry)
  yy=np.asarray([r['target'] for r in rows]);pp=np.asarray([r['prediction'] for r in rows]);bb=np.asarray([r['baseline'] for r in rows]);lab=np.asarray([r['true_low'] for r in rows]);risk=np.asarray([r['cutoff']-r['prediction'] for r in rows]);gm={}
  for group in sorted(set(r['group'] for r in rows)):
   rr=[r for r in rows if r['group']==group]
   if len(rr)>=10:
    gy=np.asarray([x['target'] for x in rr]);gp=np.asarray([x['prediction'] for x in rr]);gm[group]={'n':len(rr),'rmse':float(np.sqrt(mean_squared_error(gy,gp))),'r2':float(r2_score(gy,gp))}
  looks[str(look)]={'metrics':metric(yy,pp,lab,risk),'baseline_rmse':float(np.sqrt(mean_squared_error(yy,bb))),'rmse_delta_bootstrap95':bootstrap(rows),
   'coverage_90':float(np.mean([r['covered_90'] for r in rows])),'coverage_95':float(np.mean([r['covered_95'] for r in rows])),'per_group':gm,'macro_group_r2':float(np.mean([x['r2'] for x in gm.values()])),'worst_group_r2':float(min(x['r2'] for x in gm.values()))}
 # Persistence and field-season action metrics.
 action={fid:any(watch_by[fid].get(k,0) and watch_by[fid].get(k-1,0) for k in (2,3)) for fid in truth};ids=sorted(truth);yt=np.asarray([truth[x] for x in ids]);aa=np.asarray([action[x] for x in ids]);normal=yt==0;low=yt==1;fp=int((aa&normal).sum());tp=int((aa&low).sum());group_fpr={}
 for group in sorted(set(groups.values())):
  normals=[x for x in ids if groups[x]==group and truth[x]==0]
  if len(normals)>=10:group_fpr[group]={'normal_n':len(normals),'fp':sum(action[x] for x in normals),'fpr':float(np.mean([action[x] for x in normals]))}
 action_summary={'n':len(ids),'low_n':int(low.sum()),'normal_n':int(normal.sum()),'tp':tp,'fp':fp,'fpr':fp/max(1,int(normal.sum())),'fpr_upper95':upper_cp(fp,int(normal.sum())),
  'sensitivity':tp/max(1,int(low.sum())),'precision':tp/max(1,tp+fp),'worst_group_fpr':max([x['fpr'] for x in group_fpr.values()],default=0.0),'group_fpr':group_fpr}
 gates={'integrity':bool(len(ids)>=m['gates']['minimum_outcomes'] and np.isfinite(audit['target']).all() and len(set(ids))==len(ids)),
  'late_beats_climatology_ci':bool(looks['3']['rmse_delta_bootstrap95'][2]<0),'late_ranking':bool(looks['3']['metrics']['auroc']>=m['gates']['late_auroc'] and looks['3']['metrics']['auprc']>looks['3']['metrics']['prevalence']),
  'late_group_robustness':bool(looks['3']['macro_group_r2']>0 and looks['3']['worst_group_r2']>0),'early_curve':bool(sum(looks[str(k)]['metrics']['auroc']>=.65 for k in (1,2,3))>=2 and looks['3']['metrics']['rmse']<=looks['1']['metrics']['rmse']),
  'calibration':bool(all(looks[str(k)]['coverage_90']>=.87 and looks[str(k)]['coverage_95']>=.92 for k in (1,2,3))),
  'action_alert':bool(action_summary['fpr']<=.05 and action_summary['fpr_upper95']<=.08 and tp>=3 and action_summary['sensitivity']>=.05 and action_summary['precision']>=.5 and action_summary['worst_group_fpr']<=.15)}
 for r in flat:r['action_alert']=int(action[r['field_id']])
 with (OUT/'final_predictions.csv').open('w',encoding='utf-8-sig',newline='') as h:
  keys=sorted(set().union(*(r.keys() for r in flat)));w=csv.DictWriter(h,fieldnames=keys);w.writeheader();w.writerows(flat)
 report={'status':'PASS_FINAL' if all(gates.values()) else 'FAIL_FINAL','executed_utc':datetime.now(timezone.utc).isoformat(),'audit_fields':len(ids),'looks':looks,'action_alert':action_summary,'pass_gates':gates,
  'claim_boundary':'Field-disjoint one-shot audit; not farm-disjoint because the legacy split hashes fields.','no_retuning_permitted':True,'freeze_manifest_sha256':sha(MANIFEST)}
 result_path.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=True),encoding='utf-8');print(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=True))

if __name__=='__main__':main()
