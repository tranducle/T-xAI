from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / 'experiments' / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_delta_z_transforms_are_deterministic_and_do_not_mutate_input():
    mod = load_script('run_delta_z_bridge.py')
    rng = np.random.default_rng(7)
    phi = rng.normal(size=(8, 2381))
    original = phi.copy()
    first = mod.transform_families(phi)
    second = mod.transform_families(phi)
    assert np.array_equal(phi, original)
    assert set(first) == {'topk_redaction', 'group_omission', 'mass_retention', 'combined'}
    assert len(first['combined']) == 18
    for family in ('topk_redaction', 'group_omission', 'mass_retention'):
        assert len(first[family]) == len(second[family])
        for (label_a, a), (label_b, b) in zip(first[family], second[family]):
            assert label_a == label_b
            assert np.array_equal(a, b)
            assert np.any(a != phi)


def test_m3_structural_verifier_misses_well_formed_edit_but_authenticator_detects_it():
    mod = load_script('run_m3_evidence_channel.py')
    rng = np.random.default_rng(3)
    X = rng.normal(size=(2, 2381))
    scores = np.array([0.91, 0.83])
    phi = rng.normal(size=(2, 2381))
    key = hashlib.sha256(b'test-key').digest()
    a = mod.evidence_bundle(0, X, scores, phi, key, 20)
    b = mod.evidence_bundle(1, X, scores, phi, key, 20)
    attacked = mod.edit_score(a, b)
    assert mod.structural_verify(attacked)
    assert not mod.authenticated_verify(attacked, key)


def test_m3_channel_detects_replay_and_accepts_canonical_rewrite(tmp_path):
    mod = load_script('run_m3_evidence_channel.py')
    rng = np.random.default_rng(4)
    X = rng.normal(size=(4, 2381))
    scores = np.array([0.91, 0.83, 0.77, 0.72])
    phi = rng.normal(size=(4, 2381))
    key = hashlib.sha256(b'test-key').digest()
    bundles = [mod.evidence_bundle(i, X, scores, phi, key, 20) for i in range(4)]
    path = tmp_path / 'clean.jsonl'
    anchor = mod.write_channel(path, bundles, key)
    assert mod.verify_channel(path, key, anchor) == (True, 'ok')

    entries = mod.read_channel(path)
    replay = entries + [entries[0]]
    replay_path = tmp_path / 'replay.jsonl'
    mod.rewrite_entries(replay_path, replay)
    ok, reason = mod.verify_channel(replay_path, key, anchor)
    assert not ok
    assert reason == 'count_or_truncation'

    canonical_path = tmp_path / 'canonical.jsonl'
    mod.rewrite_entries(canonical_path, entries, pretty=True)
    assert mod.verify_channel(canonical_path, key, anchor) == (True, 'ok')


def test_m3_channel_exposes_key_compromise_boundary(tmp_path):
    mod = load_script('run_m3_evidence_channel.py')
    rng = np.random.default_rng(5)
    X = rng.normal(size=(3, 2381))
    scores = np.array([0.91, 0.83, 0.77])
    phi = rng.normal(size=(3, 2381))
    key = hashlib.sha256(b'test-key').digest()
    bundles = [mod.evidence_bundle(i, X, scores, phi, key, 20) for i in range(3)]
    path = tmp_path / 'clean.jsonl'
    mod.write_channel(path, bundles, key)
    entries = mod.read_channel(path)
    entries[0]['payload']['claims'][0]['text'] = 'attacker edit'
    retagged, attacker_anchor = mod.retag_entries(entries, key)
    attack_path = tmp_path / 'retagged.jsonl'
    mod.rewrite_entries(attack_path, retagged)
    assert mod.verify_channel(attack_path, key, attacker_anchor) == (True, 'ok')


def test_external_linear_attribution_is_deterministic_and_featurewise():
    mod = load_script('run_external_replication.py')
    X = np.array([[1.0, 2.0, -3.0], [4.0, -5.0, 6.0]])
    coef = np.array([0.5, -2.0, 3.0])
    phi1 = mod.linear_attribution(X, coef)
    phi2 = mod.linear_attribution(X, coef)
    expected = X * coef[None, :]
    assert np.array_equal(phi1, phi2)
    assert np.array_equal(phi1, expected)


def test_external_stratified_indices_are_deterministic_and_keep_both_classes():
    mod = load_script('run_external_replication.py')
    y = np.array([0] * 80 + [1] * 20)
    a = mod.stratified_indices(y, 30, seed=42)
    b = mod.stratified_indices(y, 30, seed=42)
    assert np.array_equal(a, b)
    assert len(a) == 30
    assert len(np.unique(a)) == 30
    assert set(y[a]) == {0, 1}
    assert int((y[a] == 1).sum()) == 6


def test_external_m3_summary_preserves_authenticated_detection_boundary():
    mod = load_script('run_external_replication.py')
    e11 = load_script('run_m3_evidence_channel.py')
    rng = np.random.default_rng(9)
    X = rng.normal(size=(6, 2381))
    scores = np.linspace(0.6, 0.95, 6)
    phi = rng.normal(size=(6, 2381))
    key = hashlib.sha256(b'external-test-key').digest()
    bundles = [e11.evidence_bundle(i, X, scores, phi, key, 20) for i in range(6)]
    out = mod.evaluate_m3(bundles, key)
    assert out['minimum_authenticated_detection_rate'] == 1.0
    assert out['mean_structural_detection_rate_well_formed_attacks'] == 0.0
    assert out['clean_false_reject_rate'] == 0.0
    assert out['channel_integrity']['malicious_detection_rate'] == 1.0
    assert out['channel_integrity']['benign_acceptance_rate'] == 1.0
    assert out['channel_integrity']['key_compromise_bypass'] is True
