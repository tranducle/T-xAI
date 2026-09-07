#!/usr/bin/env python3
"""Post-run result-integrity and claim-permission gates for E10/E11/E12."""
from __future__ import annotations
import json, math, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'gates'; OUT.mkdir(parents=True,exist_ok=True)


def save(name,obj):
    p=OUT/name
    p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    print(obj['gate_id'],obj['verdict'],'->',p)


def main()->int:
    p10=ROOT/'results'/'E10_delta_z_bridge.json'
    p11=ROOT/'results'/'E11_m3_evidence_channel.json'
    p12=ROOT/'results'/'E12_external_replication.json'
    required=(p10,p11,p12)
    if not all(p.exists() for p in required):
        print('missing',[p.name for p in required if not p.exists()],file=sys.stderr)
        return 2
    e10=json.loads(p10.read_text()); e11=json.loads(p11.read_text()); e12=json.loads(p12.read_text())
    failed=[]; warnings=[]
    rows10=e10['rows']; rows11=e11['attack_rows']; rows12=e12['delta_z']['rows']

    vals=[]
    for rows in (rows10,rows12):
        for r in rows:
            vals += [float(r[k]) for k in ('lhs_adv_tv','rhs_L1','ratio_L1','dobrushin','rhs_dobrushin','ratio_dobrushin','mean_Bz','expected_instability')]
    for r in rows11:
        vals += [float(r['structural_detection_rate']),float(r['authenticated_detection_rate'])]
    vals += [float(e11['boundary_control']['compromised_key_bypass_rate'])]
    if not all(math.isfinite(v) for v in vals): failed.append('finite_metrics')

    # E10 within-scope bridge.
    if len(rows10)!=12: failed.append('E10_expected_rows')
    if int(e10['gate']['metrics']['n_alerts'])!=500: failed.append('E10_official_sample')
    if not all(r['holds_L1'] and r['holds_dobrushin'] for r in rows10): failed.append('E10_bound_rows')
    post10=e10.get('scope_postconditions',{})
    if not (post10.get('input_max_abs_diff') == 0.0 and post10.get('detector_score_max_abs_diff') == 0.0 and post10.get('labels_equal') is True): failed.append('E10_actual_postconditions')
    if e10.get('reproducibility',{}).get('working_tree_dirty') is not False: failed.append('E10_clean_revision')
    max_ratio10=max(r['ratio_L1'] for r in rows10)
    if not (0.1 < max_ratio10 < 1.0): failed.append('E10_nontrivial_tightness')

    # E11 controlled M3 channel.
    if int(e11['gate']['metrics']['n_alerts'])!=500: failed.append('E11_official_sample')
    min_auth11=min(r['authenticated_detection_rate'] for r in rows11)
    if min_auth11 < .99: failed.append('E11_detection')
    if float(e11['gate']['metrics']['clean_false_reject_rate']) > .01: failed.append('E11_clean_acceptance')
    if float(e11['boundary_control']['compromised_key_bypass_rate']) < .99: failed.append('E11_boundary_control')
    channel11=e11.get('channel_integrity',{})
    if channel11.get('clean_channel_accepted') is not True: failed.append('E11_channel_clean')
    if float(channel11.get('malicious_detection_rate',0.0)) < .99: failed.append('E11_channel_malicious')
    if float(channel11.get('benign_acceptance_rate',0.0)) < .99: failed.append('E11_channel_benign')
    if channel11.get('key_compromise_bypass') is not True: failed.append('E11_channel_boundary')
    if e11.get('reproducibility',{}).get('working_tree_dirty') is not False: failed.append('E11_clean_revision')
    if e11['gate']['verdict']=='PASS_WITH_WARNINGS': warnings.extend(e11['gate'].get('warnings',[]))

    # E12 external replication across corpus, detector, and explainer.
    g12=e12.get('gate',{}); m12=g12.get('metrics',{}); post12=e12.get('delta_z',{}).get('scope_postconditions',{})
    if e12.get('status')!='official': failed.append('E12_official_status')
    if e12.get('corpus',{}).get('name')!='BODMAS': failed.append('E12_external_corpus')
    if 'SGDClassifier' not in e12.get('detector',{}).get('family',''): failed.append('E12_external_detector')
    if e12.get('explainer',{}).get('name')!='deterministic linear contribution': failed.append('E12_external_explainer')
    if int(m12.get('n_eval',0))!=500: failed.append('E12_official_sample')
    if int(m12.get('eval_benign',0))<=0 or int(m12.get('eval_malware',0))<=0: failed.append('E12_eval_class_diversity')
    if float(m12.get('test_auc',0.0))<.70: failed.append('E12_detector_auc')
    if len(rows12)!=12 or not all(r['holds_L1'] and r['holds_dobrushin'] for r in rows12): failed.append('E12_delta_z_rows')
    if not (post12.get('input_max_abs_diff') == 0.0 and post12.get('detector_score_max_abs_diff') == 0.0 and post12.get('labels_equal') is True): failed.append('E12_actual_postconditions')
    max_ratio12=max(r['ratio_L1'] for r in rows12)
    if not (0.01 < max_ratio12 < 1.0): failed.append('E12_nontrivial_tightness')
    if float(m12.get('m3_min_authenticated_detection',0.0))<.99: failed.append('E12_m3_detection')
    if float(m12.get('m3_clean_false_reject',1.0))>.01: failed.append('E12_m3_clean')
    if float(m12.get('m3_channel_malicious_detection',0.0))<.99: failed.append('E12_channel_malicious')
    if float(m12.get('m3_channel_benign_acceptance',0.0))<.99: failed.append('E12_channel_benign')
    if m12.get('m3_key_compromise_bypass') is not True: failed.append('E12_channel_boundary')
    if e12.get('reproducibility',{}).get('working_tree_dirty') is not False: failed.append('E12_clean_revision')
    if g12.get('verdict')=='PASS_WITH_WARNINGS': warnings.extend(g12.get('warnings',[]))

    verdict='FAIL_REPAIR' if failed else ('PASS_WITH_WARNINGS' if warnings else 'PASS')
    g6={
      'gate_id':'G6_acceptance_result_integrity','stage':'acceptance_upgrade_results',
      'verdict':verdict,
      'input_artifacts':['results/E10_delta_z_bridge.json','results/E11_m3_evidence_channel.json','results/E12_external_replication.json'],
      'metrics':{
        'E10_rows':len(rows10),'E10_max_ratio_L1':max_ratio10,'E10_all_L1_hold':all(r['holds_L1'] for r in rows10),'E10_all_dobrushin_hold':all(r['holds_dobrushin'] for r in rows10),'E10_postconditions_exact':bool(post10.get('input_max_abs_diff') == 0.0 and post10.get('detector_score_max_abs_diff') == 0.0 and post10.get('labels_equal') is True),'E10_git_revision':e10.get('reproducibility',{}).get('git_revision'),'E10_working_tree_dirty':e10.get('reproducibility',{}).get('working_tree_dirty'),
        'E11_attack_types':len(rows11),'E11_min_authenticated_detection_rate':min_auth11,'E11_clean_false_reject_rate':float(e11['gate']['metrics']['clean_false_reject_rate']),'E11_channel_malicious_detection_rate':channel11.get('malicious_detection_rate'),'E11_channel_benign_acceptance_rate':channel11.get('benign_acceptance_rate'),'E11_channel_key_compromise_bypass':channel11.get('key_compromise_bypass'),'E11_git_revision':e11.get('reproducibility',{}).get('git_revision'),'E11_working_tree_dirty':e11.get('reproducibility',{}).get('working_tree_dirty'),
        'E12_corpus':e12.get('corpus',{}).get('name'),'E12_detector':e12.get('detector',{}).get('family'),'E12_explainer':e12.get('explainer',{}).get('name'),'E12_test_auc':m12.get('test_auc'),'E12_eval_benign':m12.get('eval_benign'),'E12_eval_malware':m12.get('eval_malware'),'E12_max_ratio_L1':max_ratio12,'E12_delta_z_all_hold':m12.get('delta_z_all_hold'),'E12_m3_min_authenticated_detection':m12.get('m3_min_authenticated_detection'),'E12_channel_malicious_detection':m12.get('m3_channel_malicious_detection'),'E12_channel_benign_acceptance':m12.get('m3_channel_benign_acceptance'),'E12_git_revision':e12.get('reproducibility',{}).get('git_revision'),'E12_working_tree_dirty':e12.get('reproducibility',{}).get('working_tree_dirty'),
        'all_selected_metrics_finite':all(math.isfinite(v) for v in vals)},
      'pass_criteria':{
        'E10':'500 alerts, exact postconditions, 12 generic/Dobrushin rows, clean code revision',
        'E11':'500 alerts, all bundle/channel attacks detected, benign channel updates accepted, key-compromise boundary preserved, clean code revision',
        'E12':'different corpus, detector, and explainer; 500 class-diverse temporal holdout samples; AUC >=0.70; E10/E11 mechanisms replicate; clean code revision'},
      'failed_criteria':failed,'warnings':warnings,
      'next_allowed_step':'manuscript_claim_update' if not failed else 'repair_results',
      'fail_interpretation':'E10/E11/E12 outputs are incomplete, degenerate, non-reproducible, or inconsistent with their declared mechanisms.'}
    save('G6_acceptance_result_integrity.json',g6)

    allowed=[
      'On 500 EMBER-2018 alerts, 18 deterministic explanation-only transformations instantiate Delta_z while x, f(x), and labels remain fixed.',
      'All 12 EMBER family/kernel rows satisfy the implemented generic L_pi=1 and Dobrushin inequalities; this numerically instantiates the theorem within its stated scope but does not prove or generally validate the theorem.',
      'In a controlled EMBER M3 channel-integrity experiment, authenticated provenance detects all seven tested corruptions with no observed clean false rejections, whereas schema/coverage checks alone miss the six well-formed edits.',
      'Authenticator compromise bypasses the M3 integrity check by design, so evidence authenticity remains an external trust assumption.',
      'The Delta_z and controlled-M3 mechanism results replicate on BODMAS using a different SGD-logistic detector and deterministic linear-contribution explainer: 500 temporally held-out class-diverse samples, 12/12 bridge rows satisfying the implemented inequalities, and the same authenticated-channel detection and trust-anchor boundary.',
      'The external replication reduces dependence of these mechanism-level observations on a single corpus, detector family, and explainer, but does not establish broad cross-domain or operational generality.'
    ]
    forbidden=[
      'T-XAI prevents arbitrary M3 attacks or secures a deployed SOC evidence channel.',
      'The experiments prove or empirically validate the theorem in general.',
      'The actionability proxy demonstrates analyst benefit, trust, usability, or deployed response improvement.',
      'Two static-malware corpora establish cross-domain, cross-modality, or deployment generality.',
      'The BODMAS linear-contribution replication establishes superiority of T-XAI, TreeSHAP, or the linear explainer.'
    ]
    g8={
      'gate_id':'G8_acceptance_claim_permission','stage':'results_to_manuscript_claims',
      'verdict':'PASS_WITH_WARNINGS' if not failed else 'FAIL_REPAIR',
      'input_artifacts':['results/gates/G0_data_provenance.json','results/gates/G1_preprocessing_integrity.json','results/gates/G2_baseline_sanity.json','results/E10_delta_z_bridge.json','results/E11_m3_evidence_channel.json','results/E12_external_replication.json','results/gates/G6_acceptance_result_integrity.json'],
      'metrics':{'allowed_claim_count':len(allowed),'forbidden_claim_count':len(forbidden),'external_replication_dimensions':['corpus','detector','explainer'],'weakest_required_gate':'PASS_WITH_WARNINGS (controlled M3 and external replication)'},
      'pass_criteria':{'wording':'claims remain within controlled mechanism-level, two-static-malware-corpus, non-human-study boundaries'},
      'failed_criteria':failed,
      'warnings':['M3 claim must remain controlled-channel rather than deployment-wide.','Delta_z result is an in-scope numerical instantiation, not additional proof.','E12 improves external validity at the mechanism level but both corpora are static PE malware datasets.'],
      'allowed_claims':allowed,'forbidden_claims':forbidden,
      'next_allowed_step':'bounded_manuscript_revision' if not failed else 'repair_results',
      'fail_interpretation':'Manuscript wording would exceed the passed scientific evidence gates.'}
    save('G8_acceptance_claim_permission.json',g8)
    return 0 if not failed else 2

if __name__=='__main__': raise SystemExit(main())
