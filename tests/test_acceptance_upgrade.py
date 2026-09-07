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
