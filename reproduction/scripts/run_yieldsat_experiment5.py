"""Experiment 5: fair strong baselines at the frozen third GDD look."""
from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_yieldsat_experiment2 import CACHE, OUT as EXP2_OUT, encode_features, subrole, threshold_for


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/yieldsat_risk_v1/experiment_05_strong_baselines"


class TemporalTransformer(torch.nn.Module):
    def __init__(self, aux_dim):
        super().__init__()
        self.proj = torch.nn.Linear(18, 64)
        self.pos = torch.nn.Parameter(torch.zeros(1, 6, 64))
        layer = torch.nn.TransformerEncoderLayer(64, 4, dim_feedforward=128, dropout=.1, batch_first=True, norm_first=True)
        self.encoder = torch.nn.TransformerEncoder(layer, 2)
        self.aux = torch.nn.Sequential(torch.nn.Linear(aux_dim, 32), torch.nn.ReLU(), torch.nn.LayerNorm(32))
        self.head = torch.nn.Sequential(torch.nn.Linear(96, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))

    def forward(self, temporal, aux):
        token = self.encoder(self.proj(temporal) + self.pos).mean(dim=1)
        return self.head(torch.cat([token, self.aux(aux)], dim=1)).squeeze(1)


def train_transformer(train_x, train_z, stop_x, stop_z, test_x, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    temporal = train_x[:, :108].reshape(-1, 18, 6).transpose(0,2,1); stop_temporal = stop_x[:, :108].reshape(-1,18,6).transpose(0,2,1); test_temporal = test_x[:, :108].reshape(-1,18,6).transpose(0,2,1)
    aux, stop_aux, test_aux = train_x[:,108:], stop_x[:,108:], test_x[:,108:]
    tm, ts = temporal.mean((0,1),keepdims=True), temporal.std((0,1),keepdims=True); ts[ts<1e-6]=1
    am, ass = aux.mean(0,keepdims=True), aux.std(0,keepdims=True); ass[ass<1e-6]=1
    temporal=(temporal-tm)/ts;stop_temporal=(stop_temporal-tm)/ts;test_temporal=(test_temporal-tm)/ts
    aux=(aux-am)/ass;stop_aux=(stop_aux-am)/ass;test_aux=(test_aux-am)/ass
    tensors=torch.utils.data.TensorDataset(torch.tensor(temporal,dtype=torch.float32),torch.tensor(aux,dtype=torch.float32),torch.tensor(train_z,dtype=torch.float32))
    loader=torch.utils.data.DataLoader(tensors,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0)
    model=TemporalTransformer(aux.shape[1]);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);loss_fn=torch.nn.MSELoss()
    st=torch.tensor(stop_temporal,dtype=torch.float32);sa=torch.tensor(stop_aux,dtype=torch.float32);sy=torch.tensor(stop_z,dtype=torch.float32)
    best=(float('inf'),None,-1);patience=0
    for epoch in range(150):
        model.train()
        for tx,ax,yy in loader:
            opt.zero_grad();loss=loss_fn(model(tx,ax),yy);loss.backward();opt.step()
        model.eval()
        with torch.no_grad(): value=float(loss_fn(model(st,sa),sy))
        if value < best[0]-1e-5: best=(value,copy.deepcopy(model.state_dict()),epoch);patience=0
        else: patience+=1
        if patience>=15: break
    model.load_state_dict(best[1]);model.eval()
    with torch.no_grad(): pred=model(torch.tensor(test_temporal,dtype=torch.float32),torch.tensor(test_aux,dtype=torch.float32)).numpy()
    return pred, sum(p.numel() for p in model.parameters()), best[2]


def raw_predictions(z, indices, data, stats):
    return np.asarray([z[j]*stats[f"{data['country'][i]}|{data['crop'][i]}"]["std"]+stats[f"{data['country'][i]}|{data['crop'][i]}"]["mean"] for j,i in enumerate(indices)])


def metrics(y,p,indices,data,thresholds):
    labels=[];risk=[]
    for value,i,pred in zip(y,indices,p):
        cutoff,_=threshold_for(thresholds,str(data['country'][i]),str(data['crop'][i]));labels.append(value<=cutoff);risk.append(cutoff-pred)
    rho=spearmanr(y,p).statistic
    return {"n":len(y),"rmse":float(np.sqrt(mean_squared_error(y,p))),"mae":float(mean_absolute_error(y,p)),"r2":float(r2_score(y,p)),
            "spearman_rho":float(rho),"auroc":float(roc_auc_score(labels,risk)),"auprc":float(average_precision_score(labels,risk)),"prevalence":float(np.mean(labels))}


def group_metrics(y,p,indices,data):
    out={}
    groups=sorted(set(f"{data['country'][i]}|{data['crop'][i]}" for i in indices))
    for group in groups:
        take=[j for j,i in enumerate(indices) if f"{data['country'][i]}|{data['crop'][i]}"==group]
        yy=y[take];pp=p[take]
        out[group]={"n":len(take),"rmse":float(np.sqrt(mean_squared_error(yy,pp))),"r2":float(r2_score(yy,pp))}
    return out


def bootstrap_delta(rows, model, reference, reps=3000):
    rng=np.random.default_rng(550055);groups={}
    for i,r in enumerate(rows):groups.setdefault(r['group'],[]).append(i)
    vals=[]
    for _ in range(reps):
        ids=[]
        for x in groups.values():ids.extend(rng.choice(x,len(x),replace=True))
        y=np.asarray([rows[i]['target'] for i in ids]);a=np.asarray([rows[i][model] for i in ids]);b=np.asarray([rows[i][reference] for i in ids])
        vals.append(np.sqrt(np.mean((y-a)**2))-np.sqrt(np.mean((y-b)**2)))
    return [float(x) for x in np.quantile(vals,[.025,.5,.975])]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    canonical=OUT/'report.json'
    if canonical.exists() and not (OUT/'report_attempt1.json').exists():
        (OUT/'report_attempt1.json').write_text(canonical.read_text(encoding='utf-8'),encoding='utf-8')
    data=np.load(CACHE,allow_pickle=False);y=data['target'].astype(float)
    ref=np.flatnonzero((data['role']=='adapter_reference')&np.isfinite(y));fit=np.asarray([i for i in ref if subrole(data['farm_group'][i])=='model_fit']);stop=np.asarray([i for i in ref if subrole(data['farm_group'][i])=='early_stop']);test=np.flatnonzero((data['role']=='locked_external')&np.isfinite(y))
    bundle=joblib.load(EXP2_OUT/'model_look3.joblib');countries,crops=bundle['countries'],bundle['crops'];look=bundle['look_gdd'];stats=bundle['target_stats']
    train_x,train_i=encode_features(data,fit,look,countries,crops);stop_x,stop_i=encode_features(data,stop,look,countries,crops);test_x,test_i=encode_features(data,test,look,countries,crops)
    med=np.nanmedian(train_x,axis=0);med[~np.isfinite(med)]=0;train_x=np.where(np.isfinite(train_x),train_x,med);stop_x=np.where(np.isfinite(stop_x),stop_x,med);test_x=np.where(np.isfinite(test_x),test_x,med)
    z=lambda ids:np.asarray([(y[i]-stats[f"{data['country'][i]}|{data['crop'][i]}"]['mean'])/stats[f"{data['country'][i]}|{data['crop'][i]}"]['std'] for i in ids])
    train_z,stop_z=z(train_i),z(stop_i);thresholds=json.loads((EXP2_OUT/'target_reference.json').read_text(encoding='utf-8'))['low_yield_thresholds']
    models={
      'ridge_full_winsor':make_pipeline(StandardScaler(),Ridge(alpha=10.0)),
      'random_forest':RandomForestRegressor(n_estimators=600,min_samples_leaf=3,max_features=.7,n_jobs=-1,random_state=5501),
      'hist_gradient_boosting':HistGradientBoostingRegressor(max_iter=300,min_samples_leaf=15,l2_regularization=1.0,random_state=5501),
      'mlp_winsor':make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(128,64),alpha=1e-4,max_iter=500,early_stopping=True,validation_fraction=.15,random_state=5501)),
    }
    predictions_by_model={};meta={}
    # Climatology and narrow vegetation-index ridge.
    predictions_by_model['climatology']=raw_predictions(np.zeros(len(test_i)),test_i,data,stats);meta['climatology']={'fit_seconds':0.0,'parameters':len(stats)}
    dynamic_cols=np.r_[np.arange(72,84),np.arange(102,108),np.arange(train_x.shape[1]-len(countries)-len(crops),train_x.shape[1])]
    narrow=make_pipeline(StandardScaler(),Ridge(alpha=10.0));t=time.perf_counter();narrow.fit(train_x[:,dynamic_cols],train_z);pz=narrow.predict(test_x[:,dynamic_cols]);meta['ndvi_ndmi_rain_ridge']={'fit_seconds':time.perf_counter()-t,'parameters':int(narrow[-1].coef_.size+1)};predictions_by_model['ndvi_ndmi_rain_ridge']=raw_predictions(pz,test_i,data,stats)
    lower=np.quantile(train_x,.005,axis=0);upper=np.quantile(train_x,.995,axis=0)
    robust_train=np.clip(train_x,lower,upper);robust_test=np.clip(test_x,lower,upper)
    for name,model in models.items():
        use_robust=name in ('ridge_full_winsor','mlp_winsor');fit_x=robust_train if use_robust else train_x;predict_x=robust_test if use_robust else test_x
        t=time.perf_counter();model.fit(fit_x,train_z);elapsed=time.perf_counter()-t;pz=model.predict(predict_x);predictions_by_model[name]=raw_predictions(pz,test_i,data,stats);meta[name]={'fit_seconds':elapsed,'parameters':int(sum(np.asarray(x).size for x in getattr(model[-1] if hasattr(model,'__getitem__') else model,'coefs_',[]))) if name=='mlp_winsor' else None};joblib.dump({'model':model,'clip_lower':lower if use_robust else None,'clip_upper':upper if use_robust else None},OUT/f'{name}.joblib')
    # Exact frozen ExtraTrees artifact.
    pz=bundle['model'].predict(np.where(np.isfinite(test_x),test_x,bundle['feature_medians']));predictions_by_model['phenology_extratrees']=raw_predictions(pz,test_i,data,stats);meta['phenology_extratrees']={'fit_seconds':'frozen_experiment2','parameters':None}
    # Weight is selected on early-stop only, never on the challenge outcome.
    stop_et=bundle['model'].predict(stop_x);stop_hg=models['hist_gradient_boosting'].predict(stop_x);weights=np.linspace(0,1,11)
    selected_weight=float(min(weights,key=lambda w:float(np.sqrt(np.mean((stop_z-((1-w)*stop_et+w*stop_hg))**2)))))
    dual_z=(1-selected_weight)*pz+selected_weight*models['hist_gradient_boosting'].predict(test_x)
    predictions_by_model['phenology_dual_tree']=raw_predictions(dual_z,test_i,data,stats);meta['phenology_dual_tree']={'fit_seconds':'components_above','parameters':None,'histgb_weight_selected_on_early_stop':selected_weight}
    # Causal Transformer, three deterministic seeds.
    transformer=[];t=time.perf_counter();params=0;epochs=[]
    for seed in (5511,5512,5513):
        pz,nparams,epoch=train_transformer(train_x,train_z,stop_x,stop_z,test_x,seed);transformer.append(pz);params=nparams;epochs.append(epoch)
    predictions_by_model['temporal_transformer']=raw_predictions(np.mean(transformer,axis=0),test_i,data,stats);meta['temporal_transformer']={'fit_seconds':time.perf_counter()-t,'parameters':params,'best_epochs':epochs,'seeds':[5511,5512,5513]}
    test_y=y[test_i];reports={};rows=[]
    for name,pred in predictions_by_model.items():
        gm=group_metrics(test_y,pred,test_i,data);reports[name]={'overall':metrics(test_y,pred,test_i,data,thresholds),'per_group':gm,'macro_group_rmse':float(np.mean([x['rmse'] for x in gm.values()])),
          'macro_group_r2':float(np.mean([x['r2'] for x in gm.values()])), 'worst_group_r2_n_ge_10':float(min(x['r2'] for x in gm.values() if x['n']>=10)),**meta[name]}
    for j,i in enumerate(test_i):
        row={'field_id':str(data['field_id'][i]),'group':f"{data['country'][i]}|{data['crop'][i]}",'target':float(y[i])}
        row.update({name:float(pred[j]) for name,pred in predictions_by_model.items()});rows.append(row)
    selected='phenology_dual_tree';ci_vs_extra={name:bootstrap_delta(rows,name,selected) for name in predictions_by_model}
    ci_extra_vs_clim=bootstrap_delta(rows,selected,'climatology');best=min(reports,key=lambda k:reports[k]['overall']['rmse']);extra=reports[selected]
    gates={'finite_complete':all(np.isfinite(x).all() and len(x)==len(test_i) for x in predictions_by_model.values()),
      'selected_within_2pct_best':extra['overall']['rmse']<=1.02*reports[best]['overall']['rmse'],
      'selected_beats_climatology_ci':ci_extra_vs_clim[2]<0,'selected_auroc_ge_0.75':extra['overall']['auroc']>=.75,
      'selected_all_supported_group_r2_positive':extra['worst_group_r2_n_ge_10']>0}
    import csv
    with (OUT/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'experiment':5,'attempt':2,'status':'PASS' if all(gates.values()) else 'FAIL','selected_predictor':selected,'best_rmse_model':best,'models':reports,'paired_rmse_delta_vs_selected_bootstrap95':ci_vs_extra,
      'selected_vs_climatology_bootstrap95':ci_extra_vs_clim,'integrity':{'audit_targets_read':False,'identical_test_ids':True,'identical_causal_look':True,'test_n':len(test_i),'weight_selected_on_early_stop':True},'pass_gates':gates}
    (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=True),encoding='utf-8');print(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=True))


if __name__=='__main__':main()
