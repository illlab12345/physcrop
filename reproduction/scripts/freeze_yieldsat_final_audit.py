"""Freeze hashes and gates before revealing YieldSAT audit outcomes."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/yieldsat_risk_v1';OUT=BASE/'final_audit';MANIFEST=OUT/'freeze_manifest.json'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 if MANIFEST.exists():raise SystemExit('Final audit is already frozen')
 cache=BASE/'cache/audit_features_blind.npz';report=BASE/'cache/audit_features_blind.report.json';r=json.loads(report.read_text(encoding='utf-8'))
 d=np.load(cache,allow_pickle=False)
 if r['audit_targets_read'] or r['include_target'] or np.isfinite(d['target']).any():raise RuntimeError('Blind audit cache contains outcomes')
 if len(d['field_id'])<300 or len(set(map(str,d['field_id'])))!=len(d['field_id']):raise RuntimeError('Audit support/uniqueness gate failed')
 files=[OUT/'FINAL_AUDIT_PROTOCOL.md',BASE/'MASTER_PROTOCOL.md',cache,report,
        BASE/'experiment_02_early_curve/milestones.json',BASE/'experiment_02_early_curve/target_reference.json',
        *[BASE/f'experiment_02_early_curve/model_look{i}.joblib' for i in (1,2,3)],
        *[BASE/f'experiment_{i:02d}_{name}/report.json' for i,name in ((1,'official_baseline'),(2,'early_curve'),(3,'calibration'),(4,'familywise_alert'),(5,'strong_baselines'),(6,'ablations'),(7,'spatial_maps'),(8,'robustness'),(9,'anda_mechanism'))],
        ROOT/'scripts/run_yieldsat_final_audit.py']
 missing=[str(x) for x in files if not x.exists()]
 if missing:raise RuntimeError(f'Missing frozen files: {missing}')
 payload={'status':'frozen_before_outcome_reveal','created_utc':datetime.now(timezone.utc).isoformat(),'audit_fields':len(d['field_id']),'audit_targets_finite':0,
  'excluded_structural_fields':r['excluded_fields'],'field_ids_sha256':hashlib.sha256('\n'.join(sorted(map(str,d['field_id']))).encode()).hexdigest(),
  'selected_calibration':{'1':'global_raw','2':'hierarchy_power50','3':'global_raw'},'alpha_spending':[.010,.015,.025],'persistence':2,
  'gates':{'minimum_outcomes':300,'late_auroc':.70,'look_auroc_count_ge_065':2,'coverage_slack':.03,'fpr':.05,'fpr_upper95':.08,'minimum_tp':3,'minimum_sensitivity':.05,'minimum_precision':.50,'worst_group_fpr':.15,'worst_group_r2_strictly_positive':True},
  'hashes':{str(x.relative_to(ROOT)).replace('\\','/'):sha(x) for x in files}}
 OUT.mkdir(parents=True,exist_ok=True);MANIFEST.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(payload,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
