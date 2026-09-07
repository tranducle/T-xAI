#!/usr/bin/env python3
"""E11: controlled M3 evidence-channel integrity experiment.

The experiment builds real per-alert evidence bundles from the EMBER-2018
substrate and TreeSHAP outputs, then applies well-formed channel corruptions.
It compares a schema/coverage-only verifier with authenticated provenance using
HMAC-SHA256. HMAC is a reproducible stand-in for a protected signing/MAC service;
its secret is modeled as outside the attacker-controlled explanation channel.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from txai_exp.config import EMBER_GROUPS, ExperimentConfig  # noqa: E402
from txai_exp.ember.substrate import build_ember_substrate  # noqa: E402
from txai_exp.pipeline import explain  # noqa: E402

RESULTS=ROOT/'results'


def canonical(obj: dict) -> bytes:
    return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()


def tag(bundle: dict, key: bytes) -> str:
    core={k:v for k,v in bundle.items() if k!='auth_tag'}
    return hmac.new(key,canonical(core),hashlib.sha256).hexdigest()


def group_of(idx: int) -> str:
    for name,(lo,hi) in EMBER_GROUPS.items():
        if lo <= idx < hi: return name
    raise ValueError(idx)


def evidence_bundle(i: int, X: np.ndarray, scores: np.ndarray,
                    phi: np.ndarray, key: bytes, top_k: int=20) -> dict:
    idx=np.argsort(-np.abs(phi[i]))[:top_k]
    vals=phi[i,idx]
    top_group=group_of(int(idx[0]))
    explanation_payload={
        'top_k':int(top_k),
        'indices':[int(x) for x in idx],
        'values':[round(float(x),10) for x in vals],
        'top_group':top_group,
    }
    sample_fp=hashlib.sha256(np.ascontiguousarray(X[i]).view(np.uint8)).hexdigest()
    expl_digest=hashlib.sha256(canonical(explanation_payload)).hexdigest()
    score=round(float(scores[i]),10)
    bundle={
        'record_id':f'alert-{i:05d}',
        'claims':[
            {'id':'c_alert','text':'detector score is recorded for this alert'},
            {'id':'c_expl','text':f'top explanation group is {top_group}'},
            {'id':'c_release','text':'analyst receives a top-k explanation view'},
        ],
        'evidence':[
            {'id':'e_detector','sample_fingerprint':sample_fp,'detector_score':score,'predicted_alert':bool(score >= 0.5)},
            {'id':'e_explanation','explanation_digest':expl_digest,'top_group':top_group,'top_k':int(top_k)},
            {'id':'e_policy','receiver_role':'analyst','release_width':int(top_k)},
        ],
        'edges':[['c_alert','e_detector'],['c_expl','e_explanation'],['c_release','e_policy']],
    }
    bundle['auth_tag']=tag(bundle,key)
    return bundle


def structural_verify(bundle: dict) -> bool:
    try:
        claims={x['id'] for x in bundle['claims']}
        evidence={x['id'] for x in bundle['evidence']}
        if claims != {'c_alert','c_expl','c_release'}: return False
        if evidence != {'e_detector','e_explanation','e_policy'}: return False
        if len(bundle['edges']) != 3: return False
        seen_claims=set()
        for c,e in bundle['edges']:
            if c not in claims or e not in evidence: return False
            seen_claims.add(c)
        return seen_claims == claims
    except Exception:
        return False


def authenticated_verify(bundle: dict, key: bytes) -> bool:
    if not structural_verify(bundle): return False
    supplied=bundle.get('auth_tag','')
    expected=tag(bundle,key)
    return hmac.compare_digest(str(supplied),expected)


def edit_score(b:dict, other:dict)->dict:
    out=copy.deepcopy(b)
    ev=next(x for x in out['evidence'] if x['id']=='e_detector')
    ev['detector_score']=round(1.0-float(ev['detector_score']),10)
    ev['predicted_alert']=bool(ev['detector_score']>=0.5)
    return out


def substitute_explanation(b:dict, other:dict)->dict:
    out=copy.deepcopy(b)
    dst=next(x for x in out['evidence'] if x['id']=='e_explanation')
    src=next(x for x in other['evidence'] if x['id']=='e_explanation')
    dst.update({k:v for k,v in src.items() if k!='id'})
    return out


def edit_role(b:dict, other:dict)->dict:
    out=copy.deepcopy(b); ev=next(x for x in out['evidence'] if x['id']=='e_policy'); ev['receiver_role']='public'; return out


def rebind_sample(b:dict, other:dict)->dict:
    out=copy.deepcopy(b); dst=next(x for x in out['evidence'] if x['id']=='e_detector'); src=next(x for x in other['evidence'] if x['id']=='e_detector'); dst['sample_fingerprint']=src['sample_fingerprint']; return out


def edit_claim(b:dict, other:dict)->dict:
    out=copy.deepcopy(b); next(x for x in out['claims'] if x['id']=='c_expl')['text']='top explanation group is benign-looking'; return out


def rebind_edge(b:dict, other:dict)->dict:
    out=copy.deepcopy(b); out['edges']=[['c_alert','e_detector'],['c_expl','e_policy'],['c_release','e_explanation']]; return out


def delete_evidence(b:dict, other:dict)->dict:
    out=copy.deepcopy(b); out['evidence']=[x for x in out['evidence'] if x['id']!='e_explanation']; return out

ATTACKS={
    'score_edit':edit_score,
    'explanation_substitution':substitute_explanation,
    'receiver_role_edit':edit_role,
    'sample_rebinding':rebind_sample,
    'claim_text_edit':edit_claim,
    'edge_rebinding':rebind_edge,
    'evidence_deletion':delete_evidence,
}


def write_channel(path: Path, bundles: list[dict], key: bytes) -> dict:
    """Write an authenticated append-only JSONL channel and return external anchor."""
    prev='0'*64
    entries=[]
    for seq,bundle in enumerate(bundles):
        core={'seq':seq,'prev_tag':prev,'payload':bundle}
        chain_tag=hmac.new(key,canonical(core),hashlib.sha256).hexdigest()
        entry={**core,'chain_tag':chain_tag}
        entries.append(entry); prev=chain_tag
    path.write_text('\n'.join(json.dumps(e,sort_keys=True,separators=(',',':')) for e in entries)+'\n')
    return {'count':len(entries),'terminal_tag':prev}


def read_channel(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_channel(path: Path, key: bytes, anchor: dict) -> tuple[bool,str]:
    try: entries=read_channel(path)
    except Exception as exc: return False,f'parse:{exc}'
    if len(entries)!=int(anchor['count']): return False,'count_or_truncation'
    prev='0'*64
    for seq,e in enumerate(entries):
        if int(e.get('seq',-1))!=seq: return False,'sequence'
        if e.get('prev_tag')!=prev: return False,'previous_tag'
        core={'seq':seq,'prev_tag':prev,'payload':e.get('payload')}
        expected=hmac.new(key,canonical(core),hashlib.sha256).hexdigest()
        if not hmac.compare_digest(str(e.get('chain_tag','')),expected): return False,'chain_tag'
        if not authenticated_verify(e.get('payload',{}),key): return False,'bundle_auth'
        prev=expected
    if prev!=anchor['terminal_tag']: return False,'terminal_anchor'
    return True,'ok'


def rewrite_entries(path: Path, entries: list[dict], pretty: bool=False) -> None:
    if pretty:
        # JSONL with varied whitespace and insertion order, but identical parsed semantics.
        path.write_text('\n'.join(json.dumps(e,sort_keys=False,separators=(', ', ': ')) for e in entries)+'\n')
    else:
        path.write_text('\n'.join(json.dumps(e,sort_keys=True,separators=(',',':')) for e in entries)+'\n')


def retag_entries(entries: list[dict], key: bytes) -> tuple[list[dict],dict]:
    """Recompute bundle and chain tags after a compromised-key attacker edits records."""
    out=[]; prev='0'*64
    for seq,e in enumerate(entries):
        payload=copy.deepcopy(e['payload'])
        payload['auth_tag']=tag(payload,key)
        core={'seq':seq,'prev_tag':prev,'payload':payload}
        ct=hmac.new(key,canonical(core),hashlib.sha256).hexdigest()
        out.append({**core,'chain_tag':ct}); prev=ct
    return out,{'count':len(out),'terminal_tag':prev}


def channel_integrity_experiment(bundles: list[dict], key: bytes) -> dict:
    """Producer/verifier channel with replay, rollback, truncation and update controls."""
    with tempfile.TemporaryDirectory(prefix='txai_e11_') as td:
        root=Path(td); channel=root/'channel'; channel.mkdir()
        key_path=root/'verifier.key'; key_path.write_bytes(key); key_path.chmod(0o600)
        key2=key_path.read_bytes()
        base=channel/'evidence.jsonl'
        anchor=write_channel(base,bundles,key2)
        clean_ok,clean_reason=verify_channel(base,key2,anchor)
        original=read_channel(base)
        tests=[]
        def run(name,entries,expected_anchor=anchor,should_accept=False,pretty=False):
            path=channel/f'{name}.jsonl'; rewrite_entries(path,entries,pretty=pretty)
            ok,reason=verify_channel(path,key2,expected_anchor)
            tests.append({'test':name,'accepted':bool(ok),'detected':bool(not ok),'reason':reason,'expected_accept':bool(should_accept),'correct':bool(ok==should_accept)})
        # Malicious channel changes without authenticator access.
        e=copy.deepcopy(original); e[0]['payload']['claims'][0]['text']='tampered claim'; run('content_edit',e)
        e=copy.deepcopy(original); e.insert(1,copy.deepcopy(e[0])); run('insertion',e)
        e=copy.deepcopy(original[:-1]); run('truncation',e)
        e=copy.deepcopy(original); e.append(copy.deepcopy(e[0])); run('replay',e)
        e=copy.deepcopy(original); e[0],e[1]=e[1],e[0]; run('reorder',e)
        e=copy.deepcopy(original[:max(1,len(original)//2)]); run('rollback_prefix',e)
        e=copy.deepcopy(original); e[0]['chain_tag']='0'*64; run('tag_corruption',e)
        # Benign reserialization must not be rejected.
        run('canonical_rewrite',copy.deepcopy(original),anchor,should_accept=True,pretty=True)
        # Legitimate append is produced with the external key and accompanied by a new anchor.
        appended=copy.deepcopy(bundles)+[copy.deepcopy(bundles[-1])]
        appended[-1]['record_id']='legitimate-update'
        appended[-1]['auth_tag']=tag(appended[-1],key2)
        app_path=channel/'legitimate_append.jsonl'; app_anchor=write_channel(app_path,appended,key2)
        ok,reason=verify_channel(app_path,key2,app_anchor)
        tests.append({'test':'legitimate_append','accepted':bool(ok),'detected':bool(not ok),'reason':reason,'expected_accept':True,'correct':bool(ok)})
        # Compromised authenticator boundary: edit then retag full log and external anchor.
        e=copy.deepcopy(original); e[0]['payload']['claims'][0]['text']='attacker with key'
        retagged,attacker_anchor=retag_entries(e,key2)
        path=channel/'key_compromise.jsonl'; rewrite_entries(path,retagged)
        ok,reason=verify_channel(path,key2,attacker_anchor)
        tests.append({'test':'key_compromise_full_retag','accepted':bool(ok),'detected':bool(not ok),'reason':reason,'expected_accept':True,'correct':bool(ok),'boundary_control':True})
        malicious=[t for t in tests if t['test'] in {'content_edit','insertion','truncation','replay','reorder','rollback_prefix','tag_corruption'}]
        benign=[t for t in tests if t['test'] in {'canonical_rewrite','legitimate_append'}]
        boundary=next(t for t in tests if t['test']=='key_compromise_full_retag')
        return {
            'clean_channel_accepted':bool(clean_ok),'clean_reason':clean_reason,
            'external_anchor_used':True,'key_outside_channel':True,'key_file_mode':'0600',
            'tests':tests,
            'malicious_detection_rate':float(np.mean([t['detected'] for t in malicious])),
            'benign_acceptance_rate':float(np.mean([t['accepted'] for t in benign])),
            'key_compromise_bypass':bool(boundary['accepted']),
        }


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--n-alerts',type=int,default=500)
    ap.add_argument('--train-size',type=int,default=None)
    ap.add_argument('--test-size',type=int,default=50_000)
    ap.add_argument('--tag',default=None)
    args=ap.parse_args()
    t0=time.perf_counter()
    cfg=ExperimentConfig(n_explain=args.n_alerts,n_robust=args.n_alerts)
    sub,provenance=build_ember_substrate(cfg,'histgb',cfg.seed,args.n_alerts,train_size=args.train_size,test_size=args.test_size)
    X=np.asarray(sub.X_alerts[:args.n_alerts]); scores=np.asarray(sub.score_fn(X)); phi=explain(sub,cfg,'treeshap',X)
    key=hashlib.sha256(b'T-XAI-E11-controlled-test-key-seed-42').digest()
    bundles=[evidence_bundle(i,X,scores,phi,key,cfg.top_k) for i in range(len(X))]
    channel=channel_integrity_experiment(bundles,key)

    clean_struct=[structural_verify(b) for b in bundles]
    verify_times=[]; clean_auth=[]
    for b in bundles:
        s=time.perf_counter_ns(); ok=authenticated_verify(b,key); verify_times.append((time.perf_counter_ns()-s)/1e6); clean_auth.append(ok)

    rows=[]
    n=len(bundles)
    for name,fn in ATTACKS.items():
        struct_detect=0; auth_detect=0
        for i,b in enumerate(bundles):
            other=bundles[(i+1)%n]
            attacked=fn(b,other)
            struct_detect += int(not structural_verify(attacked))
            auth_detect += int(not authenticated_verify(attacked,key))
        rows.append({'attack':name,'n':n,'structural_detection_rate':struct_detect/n,'authenticated_detection_rate':auth_detect/n})

    # Boundary control: if the verifier key is compromised, a channel attacker can retag a tampered record.
    compromised_bypass=0
    for i,b in enumerate(bundles):
        attacked=edit_score(b,bundles[(i+1)%n])
        attacked['auth_tag']=tag(attacked,key)
        compromised_bypass += int(authenticated_verify(attacked,key))
    compromised_bypass_rate=compromised_bypass/n

    primary=[r for r in rows if r['attack']!='evidence_deletion']
    auth_min=min(r['authenticated_detection_rate'] for r in rows)
    struct_mean=float(np.mean([r['structural_detection_rate'] for r in primary]))
    auth_mean=float(np.mean([r['authenticated_detection_rate'] for r in primary]))
    clean_false_reject=1.0-float(np.mean(clean_auth))
    clean_struct_false_reject=1.0-float(np.mean(clean_struct))
    failed=[]
    if clean_false_reject > 0.01: failed.append('clean_false_reject')
    if auth_min < 0.99: failed.append('primary_attack_detection')
    if auth_mean <= struct_mean: failed.append('authenticated_improves_over_structural')
    if compromised_bypass_rate < 0.99: failed.append('key_compromise_boundary_control')
    if not channel['clean_channel_accepted']: failed.append('channel_clean_acceptance')
    if channel['malicious_detection_rate'] < 0.99: failed.append('channel_malicious_detection')
    if channel['benign_acceptance_rate'] < 0.99: failed.append('channel_benign_updates')
    if not channel['key_compromise_bypass']: failed.append('channel_key_compromise_boundary')
    verdict='PASS_WITH_WARNINGS' if not failed else 'FAIL_REPAIR'
    gate={
        'gate_id':'G3_m3_integrity_mechanism','stage':'M3_evidence_channel_integrity',
        'verdict':verdict,
        'input_artifacts':['EMBER-2018 alert features','TreeSHAP explanations','controlled evidence bundles'],
        'metrics':{'n_alerts':n,'attack_types':len(rows),'clean_false_reject_rate':clean_false_reject,'structural_clean_false_reject_rate':clean_struct_false_reject,'minimum_authenticated_detection_rate':auth_min,'mean_structural_detection_rate_well_formed_attacks':struct_mean,'mean_authenticated_detection_rate_well_formed_attacks':auth_mean,'compromised_key_bypass_rate':compromised_bypass_rate,'median_auth_verify_ms':float(statistics.median(verify_times)),'p95_auth_verify_ms':float(np.percentile(verify_times,95)),'channel_clean_accepted':channel['clean_channel_accepted'],'channel_malicious_detection_rate':channel['malicious_detection_rate'],'channel_benign_acceptance_rate':channel['benign_acceptance_rate'],'channel_key_compromise_bypass':channel['key_compromise_bypass']},
        'pass_criteria':{'clean_false_reject_rate':'<=0.01','primary_attack_detection':'>=0.99 for every attack type','authenticated_improves_over_structural':'mean authenticated detection > mean structural detection on well-formed edits','key_compromise_boundary_control':'>=0.99 bypass when attacker can legitimately retag, documenting authenticity assumption','channel_clean_acceptance':'clean authenticated log verifies','channel_malicious_detection':'>=0.99 across edit/insertion/truncation/replay/reorder/rollback/tag attacks','channel_benign_updates':'>=0.99 acceptance for canonical rewrite and legitimate append','channel_key_compromise_boundary':'compromised authenticator can retag and bypass'},
        'failed_criteria':failed,
        'warnings':['This is a controlled producer/verifier channel experiment on static-malware artifacts, not a live SOC compromise.','HMAC models authenticated provenance with verifier secret outside the attacked channel; key compromise is explicitly outside the protection boundary.'],
        'next_allowed_step':'bounded_C3_manuscript_claim_update' if not failed else 'repair_M3_experiment',
        'fail_interpretation':'The evidence-integrity mechanism either rejects clean records, fails to detect channel corruption, or does not expose the authenticity boundary.'
    }
    try:
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
    except Exception:
        git_revision='unknown'; git_dirty=True
    payload={'experiment':'E11_m3_evidence_channel','status':'smoke' if args.tag else 'official','purpose':'controlled M3 corruption test for claim-evidence provenance integrity','provenance':provenance,'config':{'seed':cfg.seed,'n_alerts':args.n_alerts,'train_size':args.train_size,'test_size':args.test_size,'top_k':cfg.top_k,'authenticator':'HMAC-SHA256 controlled stand-in'},'attack_rows':rows,'boundary_control':{'compromised_key_bypass_rate':compromised_bypass_rate},'channel_integrity':channel,'reproducibility':{'git_revision':git_revision,'working_tree_dirty':git_dirty,'python':sys.version.split()[0],'platform':platform.platform(),'numpy':np.__version__},'gate':gate,'elapsed_seconds':round(time.perf_counter()-t0,3)}
    suffix=f'_{args.tag}' if args.tag else ''
    out=RESULTS/f'E11_m3_evidence_channel{suffix}.json'; out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'output':str(out),'gate':gate,'elapsed_seconds':payload['elapsed_seconds']},indent=2))
    return 0 if verdict in {'PASS','PASS_WITH_WARNINGS'} else 2

if __name__=='__main__': raise SystemExit(main())
