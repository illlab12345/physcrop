"""Experiment 9: read-only synthesis of immutable Anda controlled evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/yieldsat_risk_v1/experiment_09_anda_mechanism'
OLD=ROOT/'results/protocol_v2_clean'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    stage_path=OLD/'accept_upgrade_v1/stage_gate_rescue/stage_rescue_gate_report.json'
    corrected_path=OLD/'controlled_confirmation_v4/v4_hard_episode_metric_correction_report.json'
    integrity_path=OLD/'controlled_confirmation_v4/v4_final_integrity.json'
    original_path=OLD/'controlled_confirmation_v4/v4_frozen_gate_report.json'
    stage=json.loads(stage_path.read_text(encoding='utf-8'));corrected=json.loads(corrected_path.read_text(encoding='utf-8'));integrity=json.loads(integrity_path.read_text(encoding='utf-8'));original=json.loads(original_path.read_text(encoding='utf-8'))
    idx={x['system']:x for x in corrected['summaries']};full=idx['full_physcrop'];base=idx[corrected['strongest_non_physcrop']]
    positive=[f'tpr_{x}_mean' for x in ('irregular_canopy','striped_water','field_edge_mixed','index_independent')]
    hard=[f'hard_alert_{x}_mean' for x in ('registration_shift','uniform_radiometry','thin_cloud','cast_shadow')]
    selected=stage['selected'];cont0=next(x for x in stage['contamination_summary'] if x['contamination_rate']==0.0);cont5=next(x for x in stage['contamination_summary'] if x['contamination_rate']==0.05)
    gates={
      'immutable_integrity':bool(integrity['all_integrity_checks_passed']),
      'stage_rescue_passed':bool(stage['gate_passed'] and selected['worst_stage_relative_reduction']>=.20),
      'causal_fpr_below_10':bool(next(x for x in stage['causal_summary'] if x['mode']=='past_only')['actual_fpr_mean']<10),
      'contamination_5pct_tpr_loss_le_3pp':bool(cont0['tpr_mean']-cont5['tpr_mean']<=3),
      'v4_tpr_gain_ge_2pp':bool(full['tpr_mean']-base['tpr_mean']>=2),
      'v4_pauc_noninferior':bool(full['pauc_10fpr_mean']-base['pauc_10fpr_mean']>=-1),
      'v4_clean_fpr_below_10':bool(full['clean_fpr_mean']<10),
      'v4_gain_three_positive_families':bool(sum(full[x]>base[x] for x in positive)>=3),
      'v4_all_corrected_hard_families_below_10':bool(all(full[x]<10 for x in hard)),
    }
    report={
      'experiment':9,'status':'PASS_MECHANISM_ONLY' if all(gates.values()) else 'FAIL',
      'confirmatory_status':'retrospective_read_only_not_new_confirmation',
      'mechanism_metrics':{
        'stage_worst_fpr_relative_reduction':selected['worst_stage_relative_reduction'],
        'past_only_actual_fpr':next(x for x in stage['causal_summary'] if x['mode']=='past_only')['actual_fpr_mean'],
        'contamination_5pct_tpr_loss_pp':cont0['tpr_mean']-cont5['tpr_mean'],
        'v4_tpr_full':full['tpr_mean'],'v4_tpr_baseline':base['tpr_mean'],'v4_tpr_gain_pp':full['tpr_mean']-base['tpr_mean'],
        'v4_pauc_delta_pp':full['pauc_10fpr_mean']-base['pauc_10fpr_mean'],'v4_clean_fpr':full['clean_fpr_mean'],
        'positive_families_improved':sum(full[x]>base[x] for x in positive),
        'corrected_hard_alert_rates':{x.removeprefix('hard_alert_').removesuffix('_mean'):full[x] for x in hard},
        'two_consecutive_event_recall_full':full['event_recall_two_consecutive_mean'],
        'two_consecutive_event_recall_baseline':base['event_recall_two_consecutive_mean'],
      },
      'mechanism_pass_gates':gates,
      'retained_old_strict_gate':{'gate_passed':bool(corrected['gate_passed']),'hard_no_worse_than_baseline':bool(corrected['checks']['hard_no_worse_than_baseline']),'original_gate_passed':bool(original['gate_passed'])},
      'claim_boundary':'Supports mechanisms only; not YieldSAT external yield validation or strict OOD dominance.',
      'source_hashes':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in (stage_path,corrected_path,integrity_path,original_path)},
    }
    (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
