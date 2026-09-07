#!/usr/bin/env python3
"""Post-run result-integrity and claim-permission gates for E10/E11."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'gates'; OUT.mkdir(parents=True,exist_ok=True)


def save(name,obj):
    p=OUT/name; p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n'); print(obj['gate_id'],obj['verdict'],'->',p)


def main()->int:
    p10=ROOT/'results'/'E10_delta_z_bridge.json'
    p11=ROOT/'results'/'E11_m3_evidence_channel.json'
    if not p10.exists() or not p11.exists():
        missing=[str(p) for p in (p10,p11) if not p.exists()]
        print('missing',missing,file=sys.stderr); return 2
    e10=json.loads(p10.read_text()); e11=json.loads(p11.read_text())
    failed=[]; warnings=[]
    rows10=e10['rows']; rows11=e11['attack_rows']
    vals=[]
    for r in rows10:
        vals += [float(r[k]) for k in ('lhs_adv_tv','rhs_L1','ratio_L1','dobrushin','rhs_dobrushin','ratio_dobrushin','mean_Bz','expected_instability')]
    for r in rows11:
        vals += [float(r['structural_detection_rate']),float(r['authenticated_detection_rate'])]
    vals += [float(e11['boundary_control']['compromised_key_bypass_rate'])]
    if not all(math.isfinite(v) for v in vals): failed.append('finite_metrics')
    if len(rows10)!=12: failed.append('E10_expected_rows')
    if int(e10['gate']['metrics']['n_alerts'])!=500: failed.append('E10_official_sample')
    if not all(r['holds_L1'] and r['holds_dobrushin'] for r in rows10): failed.append('E10_bound_rows')
    post=e10.get('scope_postconditions',{})
    if not (post.get('input_max_abs_diff') == 0.0 and post.get('detector_score_max_abs_diff') == 0.0 and post.get('labels_equal') is True): failed.append('E10_actual_postconditions')
    if e10.get('reproducibility',{}).get('working_tree_dirty') is not False: failed.append('E10_clean_revision')
    max_ratio=max(r['ratio_L1'] for r in rows10)
    if not (0.1 < max_ratio < 1.0): failed.append('E10_nontrivial_tightness')
    if int(e11['gate']['metrics']['n_alerts'])!=500: failed.append('E11_official_sample')
    min_auth=min(r['authenticated_detection_rate'] for r in rows11)
    if min_auth < .99: failed.append('E11_detection')
    if float(e11['gate']['metrics']['clean_false_reject_rate']) > .01: failed.append('E11_clean_acceptance')
    if float(e11['boundary_control']['compromised_key_bypass_rate']) < .99: failed.append('E11_boundary_control')
    channel=e11.get('channel_integrity',{})
    if channel.get('clean_channel_accepted') is not True: failed.append('E11_channel_clean')
    if float(channel.get('malicious_detection_rate',0.0)) < .99: failed.append('E11_channel_malicious')
    if float(channel.get('benign_acceptance_rate',0.0)) < .99: failed.append('E11_channel_benign')
    if channel.get('key_compromise_bypass') is not True: failed.append('E11_channel_boundary')
    if e11.get('reproducibility',{}).get('working_tree_dirty') is not False: failed.append('E11_clean_revision')
    if e11['gate']['verdict']=='PASS_WITH_WARNINGS': warnings.extend(e11['gate'].get('warnings',[]))
    verdict='FAIL_REPAIR' if failed else ('PASS_WITH_WARNINGS' if warnings else 'PASS')
    g6={
      'gate_id':'G6_acceptance_result_integrity','stage':'acceptance_upgrade_results',
      'verdict':verdict,'input_artifacts':[str(p10),str(p11)],
      'metrics':{'E10_rows':len(rows10),'E10_max_ratio_L1':max_ratio,'E10_all_L1_hold':all(r['holds_L1'] for r in rows10),'E10_all_dobrushin_hold':all(r['holds_dobrushin'] for r in rows10),'E10_postconditions_exact':bool(post.get('input_max_abs_diff') == 0.0 and post.get('detector_score_max_abs_diff') == 0.0 and post.get('labels_equal') is True),'E10_git_revision':e10.get('reproducibility',{}).get('git_revision'),'E10_working_tree_dirty':e10.get('reproducibility',{}).get('working_tree_dirty'),'E11_attack_types':len(rows11),'E11_min_authenticated_detection_rate':min_auth,'E11_clean_false_reject_rate':float(e11['gate']['metrics']['clean_false_reject_rate']),'E11_compromised_key_bypass_rate':float(e11['boundary_control']['compromised_key_bypass_rate']),'E11_channel_clean':channel.get('clean_channel_accepted'),'E11_channel_malicious_detection_rate':channel.get('malicious_detection_rate'),'E11_channel_benign_acceptance_rate':channel.get('benign_acceptance_rate'),'E11_channel_key_compromise_bypass':channel.get('key_compromise_bypass'),'E11_git_revision':e11.get('reproducibility',{}).get('git_revision'),'E11_working_tree_dirty':e11.get('reproducibility',{}).get('working_tree_dirty'),'all_selected_metrics_finite':all(math.isfinite(v) for v in vals)},
      'pass_criteria':{'E10_official_sample':500,'E10_bound_rows':'all 12 rows satisfy generic and Dobrushin inequalities','E10_nontrivial_tightness':'0.1 < max LHS/RHS < 1','E10_actual_postconditions':'input, scores, and labels exactly unchanged','E10_clean_revision':'working tree clean at experiment start','E11_official_sample':500,'E11_detection':'>=0.99 for all tested bundle attacks','E11_clean_acceptance':'false reject <=0.01','E11_boundary_control':'>=0.99 bypass under authenticator compromise','E11_channel_clean':'clean persistent channel accepted','E11_channel_malicious':'>=0.99 for replay/rollback/insertion/truncation/reorder/edit/tag corruption','E11_channel_benign':'>=0.99 for canonical rewrite and legitimate append','E11_channel_boundary':'key compromise bypasses as declared','E11_clean_revision':'working tree clean at experiment start'},
      'failed_criteria':failed,'warnings':warnings,
      'next_allowed_step':'manuscript_claim_update' if not failed else 'repair_results',
      'fail_interpretation':'E10/E11 outputs are incomplete, degenerate, numerically invalid, or inconsistent with their declared mechanism.'}
    save('G6_acceptance_result_integrity.json',g6)

    # G8 claim permission is deliberately bounded by the weakest gate (E11 PASS_WITH_WARNINGS).
    allowed=[
      'On 500 EMBER-2018 alerts, 18 deterministic explanation-only transformations instantiate Delta_z while x, f(x), and labels remain fixed.',
      'All 12 family/kernel rows satisfy the implemented generic L_pi=1 and Dobrushin inequalities; this numerically instantiates the theorem within its stated scope but does not prove or generally validate the theorem.',
      'In a controlled M3 channel-integrity experiment, authenticated provenance detects all seven tested corruptions with no clean false rejections, whereas schema/coverage checks alone miss the well-formed edits.',
      'Authenticator compromise bypasses the M3 integrity check by design, so evidence authenticity remains an external trust assumption.'
    ]
    forbidden=[
      'T-XAI prevents arbitrary M3 attacks or secures a deployed SOC evidence channel.',
      'The experiments prove or empirically validate the theorem in general.',
      'The actionability proxy demonstrates analyst benefit, trust, usability, or deployed response improvement.',
      'The one-corpus experiments establish cross-domain generality.'
    ]
    g8={
      'gate_id':'G8_acceptance_claim_permission','stage':'results_to_manuscript_claims',
      'verdict':'PASS_WITH_WARNINGS' if not failed else 'FAIL_REPAIR',
      'input_artifacts':[str(OUT/'G0_data_provenance.json'),str(OUT/'G1_preprocessing_integrity.json'),str(OUT/'G2_baseline_sanity.json'),str(p10),str(p11),str(OUT/'G6_acceptance_result_integrity.json')],
      'metrics':{'allowed_claim_count':len(allowed),'forbidden_claim_count':len(forbidden),'weakest_required_gate':'PASS_WITH_WARNINGS (controlled M3)'},
      'pass_criteria':{'wording':'claims remain within controlled, one-corpus, non-human-study boundaries'},
      'failed_criteria':failed,'warnings':['M3 claim must remain controlled-channel rather than deployment-wide.','Delta_z result is an in-scope numerical instantiation, not additional proof.'],
      'allowed_claims':allowed,'forbidden_claims':forbidden,
      'next_allowed_step':'bounded_manuscript_revision' if not failed else 'repair_results',
      'fail_interpretation':'Manuscript wording would exceed the passed scientific evidence gates.'}
    save('G8_acceptance_claim_permission.json',g8)
    return 0 if not failed else 2

if __name__=='__main__': raise SystemExit(main())
