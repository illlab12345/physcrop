"""Post-audit P0-B: clean null, uniform baselines, and farm-robust evidence."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import beta
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_yieldsat_experiment2 import encode_features, group_stats, subrole, threshold_for


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/yieldsat_risk_v1"
EXP2 = BASE / "experiment_02_early_curve"
OUT = BASE / "post_audit_p0p1"
ALPHAS = {1: 0.010, 2: 0.015, 3: 0.025}
MODELS = ("climatology", "ndvi_ndmi_rain_ridge", "extratrees", "hist_gradient_boosting", "dual_tree")


def cp95(x, n):
    if n == 0: return None
    return [0.0 if x == 0 else float(beta.ppf(.025, x, n-x+1)),
            1.0 if x == n else float(beta.ppf(.975, x+1, n-x))]


def raw_from_z(z, ids, data, stats):
    ans = []
    for value, i in zip(z, ids):
        s = stats[f"{data['country'][i]}|{data['crop'][i]}"]
        ans.append(float(value * s["std"] + s["mean"]))
    return np.asarray(ans)


def feature_median(train_x):
    med = np.nanmedian(train_x, axis=0); med[~np.isfinite(med)] = 0.0
    return med


def encoded(bundle, data, ids):
    x, kept = encode_features(data, ids, bundle["look_gdd"], bundle["countries"], bundle["crops"])
    return np.where(np.isfinite(x), x, bundle["feature_medians"]), kept


def metric(y, p, labels, risk, baseline):
    rmse = float(np.sqrt(mean_squared_error(y, p))); brmse = float(np.sqrt(mean_squared_error(y, baseline)))
    return {"n": len(y), "rmse": rmse, "r2": float(r2_score(y, p)),
            "nrmse_by_target_sd": float(rmse / np.std(y)), "climatology_rmse": brmse,
            "rmse_skill": float(1.0-rmse/brmse), "mse_skill": float(1.0-rmse**2/brmse**2),
            "prevalence": float(np.mean(labels)), "auroc": float(roc_auc_score(labels, risk)),
            "auprc": float(average_precision_score(labels, risk))}


def policy_summary(truth, alerted):
    ids = sorted(truth); y = np.asarray([truth[x] for x in ids], bool); a = np.asarray([alerted.get(x, False) for x in ids], bool)
    tp=int(np.sum(y&a)); fp=int(np.sum(~y&a)); fn=int(np.sum(y&~a)); tn=int(np.sum(~y&~a))
    return {"n":len(ids),"low_n":int(y.sum()),"normal_n":int((~y).sum()),"tp":tp,"fp":fp,"fn":fn,"tn":tn,
            "fpr":fp/max(1,fp+tn),"fpr_cp95":cp95(fp,fp+tn),"sensitivity":tp/max(1,tp+fn),
            "sensitivity_cp95":cp95(tp,tp+fn),"precision":tp/max(1,tp+fp),"precision_cp95":cp95(tp,tp+fp)}


def farm_bootstrap(truth, alerted, farm_by_id, reps=10000):
    farms = sorted(set(farm_by_id.values())); fields = {f:[x for x in truth if farm_by_id[x]==f] for f in farms}
    rng=np.random.default_rng(20260801); vals={"fpr":[],"sensitivity":[],"precision":[]}
    for _ in range(reps):
        sampled=rng.choice(farms,len(farms),replace=True); yy=[];aa=[]
        for farm in sampled:
            for fid in fields[str(farm)]: yy.append(bool(truth[fid]));aa.append(bool(alerted.get(fid,False)))
        y=np.asarray(yy,bool);a=np.asarray(aa,bool); tp=np.sum(y&a);fp=np.sum(~y&a)
        if np.sum(~y): vals["fpr"].append(float(fp/np.sum(~y)))
        if np.sum(y): vals["sensitivity"].append(float(tp/np.sum(y)))
        if tp+fp: vals["precision"].append(float(tp/(tp+fp)))
    return {k:{"valid_replicates":len(v),"q025_median_q975":[float(x) for x in np.quantile(v,[.025,.5,.975])]} for k,v in vals.items()}


def load_audit_truth():
    with (BASE/"final_audit/final_predictions.csv").open(encoding="utf-8-sig",newline="") as h: rows=list(csv.DictReader(h))
    truth={};target={}
    for r in rows: truth[r["field_id"]]=int(r["true_low"]);target[r["field_id"]]=float(r["target"])
    return truth,target


def fit_ridge_for_look(bundle, dev, fit):
    x, ids=encode_features(dev,fit,bundle["look_gdd"],bundle["countries"],bundle["crops"]);med=feature_median(x);x=np.where(np.isfinite(x),x,med)
    ncat=len(bundle["countries"])+len(bundle["crops"]); cols=np.r_[np.arange(72,84),np.arange(102,108),np.arange(x.shape[1]-ncat,x.shape[1])]
    z=np.asarray([(dev["target"][i]-bundle["target_stats"][f"{dev['country'][i]}|{dev['crop'][i]}"]["mean"])/bundle["target_stats"][f"{dev['country'][i]}|{dev['crop'][i]}"]["std"] for i in ids])
    model=make_pipeline(StandardScaler(),Ridge(alpha=10.0));model.fit(x[:,cols],z)
    return model,med,cols


def predictions_by_model(bundle, data, ids, ridge_pack=None):
    x,kept=encoded(bundle,data,ids); stats=bundle["target_stats"]
    etz=bundle["model"].predict(x); hgz=bundle["hist_model"].predict(x); w=float(bundle["hist_weight"])
    out={"extratrees":raw_from_z(etz,kept,data,stats),"hist_gradient_boosting":raw_from_z(hgz,kept,data,stats),
         "dual_tree":raw_from_z((1-w)*etz+w*hgz,kept,data,stats),
         "climatology":np.asarray([stats[f"{data['country'][i]}|{data['crop'][i]}"]["mean"] for i in kept])}
    if ridge_pack is not None:
        model,med,cols=ridge_pack
        xx,_=encode_features(data,ids,bundle["look_gdd"],bundle["countries"],bundle["crops"]);xx=np.where(np.isfinite(xx),xx,med)
        out["ndvi_ndmi_rain_ridge"]=raw_from_z(model.predict(xx[:,cols]),kept,data,stats)
    return kept,out


def audit_uniform_comparison(dev,audit,truth,target):
    y=dev["target"].astype(float);ref=np.flatnonzero((dev["role"]=="adapter_reference")&np.isfinite(y))
    fit=np.asarray([i for i in ref if subrole(dev["farm_group"][i])=="model_fit"]);cal=np.asarray([i for i in ref if subrole(dev["farm_group"][i])=="conformal"])
    if any(subrole(dev["farm_group"][i])!="conformal" for i in cal): raise RuntimeError("Dirty null role")
    thresholds=json.loads((EXP2/"target_reference.json").read_text(encoding="utf-8"))["low_yield_thresholds"]
    aids=np.arange(len(audit["field_id"])); watches={m:{f:{} for f in truth} for m in MODELS}; look_metrics={m:{} for m in MODELS}; null_counts={}
    prediction_rows=[]
    for look in (1,2,3):
        bundle=joblib.load(EXP2/f"model_look{look}.joblib");ridge=fit_ridge_for_look(bundle,dev,fit)
        cids,cp=predictions_by_model(bundle,dev,cal,ridge);eids,ep=predictions_by_model(bundle,audit,aids,ridge)
        if set(ep)!=set(MODELS): raise RuntimeError("Baseline set mismatch")
        null_counts[str(look)]={}
        for model in MODELS:
            null=[]
            for i,p in zip(cids,cp[model]):
                cut,_=threshold_for(thresholds,str(dev["country"][i]),str(dev["crop"][i]))
                if y[i]>cut:
                    s=bundle["target_stats"][f"{dev['country'][i]}|{dev['crop'][i]}"]["std"];null.append((cut-p)/s)
            null=np.asarray(null);null_counts[str(look)][model]={"eligible_conformal":len(cids),"non_low_null":len(null)}
            metric_rows=[]
            for i,p in zip(eids,ep[model]):
                fid=str(audit["field_id"][i]);group=f"{audit['country'][i]}|{audit['crop'][i]}";cut,_=threshold_for(thresholds,str(audit["country"][i]),str(audit["crop"][i]));s=bundle["target_stats"][group]["std"]
                score=(cut-p)/s;pval=float((1+np.sum(null>=score))/(len(null)+1));hit=bool(pval<=ALPHAS[look]);watches[model][fid][look]=hit
                metric_rows.append((fid,p,cut));prediction_rows.append({"model":model,"look":look,"field_id":fid,"prediction":float(p),"cutoff":float(cut),"pvalue":pval,"watch":int(hit)})
            yy=np.asarray([target[f] for f,_,_ in metric_rows]);pp=np.asarray([p for _,p,_ in metric_rows]);cuts=np.asarray([c for _,_,c in metric_rows]);lab=np.asarray([truth[f] for f,_,_ in metric_rows]);base=np.asarray([bundle["target_stats"][f"{audit['country'][i]}|{audit['crop'][i]}"]["mean"] for i in eids])
            look_metrics[model][str(look)]=metric(yy,pp,lab,cuts-pp,base)
    farm_by_id={str(f):str(g) for f,g in zip(audit["field_id"],audit["farm_group"])};policies={}
    for model in MODELS:
        watch={f:any(watches[model][f].values()) for f in truth}; action={f:any(watches[model][f].get(k,False) and watches[model][f].get(k-1,False) for k in (2,3)) for f in truth}
        policies[model]={"look_metrics":look_metrics[model],"watch":policy_summary(truth,watch),"action":policy_summary(truth,action),
                         "watch_farm_cluster_bootstrap95":farm_bootstrap(truth,watch,farm_by_id),"action_farm_cluster_bootstrap95":farm_bootstrap(truth,action,farm_by_id)}
    with (OUT/"p0b_uniform_audit_predictions.csv").open("w",encoding="utf-8-sig",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(prediction_rows[0]));w.writeheader();w.writerows(prediction_rows)
    return policies,null_counts,len(cal),len(set(map(str,dev["farm_group"][cal])))


def fold_bucket(farm):
    return int(hashlib.sha256(("p0b-oof-stop:"+str(farm)).encode()).hexdigest()[:12],16)%100


def farm_oof(data):
    y=data["target"].astype(float);all_ids=np.flatnonzero(np.isfinite(y));groups=np.asarray(data["farm_group"][all_ids]);splitter=GroupKFold(n_splits=5);rows=[];folds=[]
    for fold,(trpos,tepos) in enumerate(splitter.split(all_ids,groups=groups),1):
        train=all_ids[trpos];test=all_ids[tepos];fit=np.asarray([i for i in train if fold_bucket(data["farm_group"][i])<85]);stop=np.asarray([i for i in train if fold_bucket(data["farm_group"][i])>=85])
        countries=sorted(set(map(str,data["country"][fit])));crops=sorted(set(map(str,data["crop"][fit])));milestones={}
        for crop in crops:
            maxima=[float(np.nanmax(data["dynamic"][i,:data["lengths"][i],17])) for i in fit if str(data["crop"][i])==crop];milestones[crop]=float(np.median(maxima)*.65)
        stats,thresholds=group_stats(data,fit);fx,fi=encode_features(data,fit,milestones,countries,crops);sx,si=encode_features(data,stop,milestones,countries,crops);tx,ti=encode_features(data,test,milestones,countries,crops)
        med=feature_median(fx);fx=np.where(np.isfinite(fx),fx,med);sx=np.where(np.isfinite(sx),sx,med);tx=np.where(np.isfinite(tx),tx,med)
        z=lambda ids:np.asarray([(y[i]-stats[f"{data['country'][i]}|{data['crop'][i]}"]["mean"])/stats[f"{data['country'][i]}|{data['crop'][i]}"]["std"] for i in ids]);fz=z(fi);sz=z(si)
        et=ExtraTreesRegressor(n_estimators=600,min_samples_leaf=3,max_features=.7,bootstrap=False,random_state=2201,n_jobs=-1).fit(fx,fz);hg=HistGradientBoostingRegressor(max_iter=300,min_samples_leaf=15,l2_regularization=1.0,random_state=5501).fit(fx,fz)
        se,sh=et.predict(sx),hg.predict(sx);weights=np.linspace(0,1,11);weight=float(min(weights,key=lambda w:np.sqrt(np.mean((sz-((1-w)*se+w*sh))**2))))
        pred=raw_from_z((1-weight)*et.predict(tx)+weight*hg.predict(tx),ti,data,stats)
        for i,p in zip(ti,pred):
            cut,_=threshold_for(thresholds,str(data["country"][i]),str(data["crop"][i]));rows.append({"fold":fold,"field_id":str(data["field_id"][i]),"farm":str(data["farm_group"][i]),"group":f"{data['country'][i]}|{data['crop'][i]}","target":float(y[i]),"prediction":float(p),"baseline":float(stats[f"{data['country'][i]}|{data['crop'][i]}"]["mean"]),"cutoff":float(cut),"low":int(y[i]<=cut)})
        folds.append({"fold":fold,"train_fields":len(fi),"stop_fields":len(si),"test_fields":len(ti),"train_farms":len(set(map(str,data["farm_group"][fi]))),"test_farms":len(set(map(str,data["farm_group"][ti]))),"farm_overlap":len(set(map(str,data["farm_group"][train]))&set(map(str,data["farm_group"][test]))),"hist_weight":weight})
    yy=np.asarray([r["target"] for r in rows]);pp=np.asarray([r["prediction"] for r in rows]);bb=np.asarray([r["baseline"] for r in rows]);lab=np.asarray([r["low"] for r in rows]);cut=np.asarray([r["cutoff"] for r in rows]);overall=metric(yy,pp,lab,cut-pp,bb);gm={}
    for group in sorted(set(r["group"] for r in rows)):
        rr=[r for r in rows if r["group"]==group];gy=np.asarray([r["target"] for r in rr]);gp=np.asarray([r["prediction"] for r in rr]);gm[group]={"n":len(rr),"rmse":float(np.sqrt(mean_squared_error(gy,gp))),"r2":float(r2_score(gy,gp))}
    overall.update({"fields":len(rows),"farms":len(set(r["farm"] for r in rows)),"macro_group_r2":float(np.mean([v["r2"] for v in gm.values()])),"worst_group_r2":float(min(v["r2"] for v in gm.values())),"per_group":gm})
    with (OUT/"p0b_farm_oof_predictions.csv").open("w",encoding="utf-8-sig",newline="") as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return overall,folds


def main():
    OUT.mkdir(parents=True,exist_ok=True);a=np.load(BASE/"cache/development_sequences.npz",allow_pickle=False);dev={k:a[k] for k in a.files};a.close();a=np.load(BASE/"cache/audit_features_blind.npz",allow_pickle=False);audit={k:a[k] for k in a.files};a.close();truth,target=load_audit_truth()
    if set(truth)!=set(map(str,audit["field_id"])):raise RuntimeError("Audit ID mismatch")
    policies,counts,cal_n,cal_farms=audit_uniform_comparison(dev,audit,truth,target);oof,folds=farm_oof(dev)
    report={"analysis_status":"post-audit protocol correction/comparison; original frozen audit unchanged","null_definition":{"roles":["conformal"],"early_stop_excluded":True,"model_fit_excluded":True,"low_yield_excluded":True,"conformal_fields":cal_n,"conformal_farms":cal_farms,"counts":counts},"audit_uniform_policy_comparison":policies,"full_development_farm_grouped_oof":{"overall":oof,"folds":folds},"integrity":{"audit_ids_match":True,"audit_labels_used_for_fit_calibration_or_threshold_selection":False,"frozen_components_used":["extratrees","hist_gradient_boosting","dual_tree"],"post_audit_fixed_training":["ndvi_ndmi_rain_ridge"],"farm_oof_overlap_each_fold_zero":all(x["farm_overlap"]==0 for x in folds)}}
    (OUT/"p0b_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8");print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=="__main__":main()
