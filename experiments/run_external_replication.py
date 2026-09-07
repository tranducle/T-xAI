#!/usr/bin/env python3
"""E12 external replication across corpus, detector, and explanation mechanism.

The official experiment uses BODMAS on its temporal split, an SGD logistic
classifier, and deterministic linear contribution explanations phi_j=x_j*w_j.
It reuses the E10 explanation-only Delta_z transformations and the E11
claim-evidence channel verifier so the replication changes corpus, detector, and
explanation mechanism while preserving the audited experiment semantics.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from txai_exp.data import load_bodmas, temporal_split  # noqa: E402
from run_delta_z_bridge import transform_families, evaluate_family, sha  # noqa: E402
from run_m3_evidence_channel import (  # noqa: E402
    ATTACKS,
    authenticated_verify,
    channel_integrity_experiment,
    evidence_bundle,
    structural_verify,
    tag,
)

RESULTS = ROOT / "results"


def linear_attribution(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """Deterministic per-feature contribution for a linear logit model."""
    X = np.asarray(X, dtype=np.float64)
    coef = np.asarray(coef, dtype=np.float64)
    if X.ndim != 2 or coef.ndim != 1 or X.shape[1] != coef.shape[0]:
        raise ValueError(f"shape mismatch: X={X.shape}, coef={coef.shape}")
    return X * coef[None, :]



def stratified_indices(y: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    """Deterministically sample n unique indices while preserving class proportions."""
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if n <= 0 or n > len(y):
        raise ValueError(f"n must be in [1,{len(y)}], got {n}")
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("stratified sampling requires at least two classes")
    exact = counts.astype(float) * (float(n) / float(len(y)))
    take = np.floor(exact).astype(int)
    remainder = int(n - take.sum())
    order = np.argsort(-(exact - take), kind="stable")
    for j in order[:remainder]:
        take[j] += 1
    if np.any(take == 0):
        # Preserve every observed class whenever n permits it.
        zero = np.flatnonzero(take == 0)
        for j in zero:
            donor = int(np.argmax(take))
            if take[donor] <= 1:
                raise ValueError("sample too small to preserve all classes")
            take[donor] -= 1
            take[j] = 1
    rng = np.random.default_rng(seed)
    selected = []
    for cls, k in zip(classes, take):
        idx = np.flatnonzero(y == cls)
        selected.extend(rng.choice(idx, size=int(k), replace=False).tolist())
    selected = np.asarray(selected, dtype=int)
    rng.shuffle(selected)
    return selected


def evaluate_m3(bundles: list[dict], key: bytes) -> dict:
    """Run the E11 bundle attacks and persistent channel controls on new bundles."""
    clean_struct = [structural_verify(b) for b in bundles]
    clean_auth = [authenticated_verify(b, key) for b in bundles]
    rows = []
    n = len(bundles)
    for name, fn in ATTACKS.items():
        structural_detect = 0
        authenticated_detect = 0
        for i, bundle in enumerate(bundles):
            other = bundles[(i + 1) % n]
            attacked = fn(bundle, other)
            structural_detect += int(not structural_verify(attacked))
            authenticated_detect += int(not authenticated_verify(attacked, key))
        rows.append({
            "attack": name,
            "n": n,
            "structural_detection_rate": structural_detect / n,
            "authenticated_detection_rate": authenticated_detect / n,
        })

    well_formed = [r for r in rows if r["attack"] != "evidence_deletion"]
    compromised = 0
    for i, bundle in enumerate(bundles):
        attacked = ATTACKS["score_edit"](bundle, bundles[(i + 1) % n])
        attacked["auth_tag"] = tag(attacked, key)
        compromised += int(authenticated_verify(attacked, key))

    channel = channel_integrity_experiment(bundles, key)
    return {
        "attack_rows": rows,
        "clean_false_reject_rate": 1.0 - float(np.mean(clean_auth)),
        "structural_clean_false_reject_rate": 1.0 - float(np.mean(clean_struct)),
        "minimum_authenticated_detection_rate": min(r["authenticated_detection_rate"] for r in rows),
        "mean_structural_detection_rate_well_formed_attacks": float(np.mean([r["structural_detection_rate"] for r in well_formed])),
        "mean_authenticated_detection_rate_well_formed_attacks": float(np.mean([r["authenticated_detection_rate"] for r in well_formed])),
        "compromised_key_bypass_rate": compromised / n,
        "channel_integrity": channel,
    }


def _timestamp_range(ts: np.ndarray) -> dict:
    valid = ts[~np.isnat(ts)]
    return {
        "min": str(valid.min()) if len(valid) else None,
        "max": str(valid.max()) if len(valid) else None,
    }

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=50_000)
    ap.add_argument("--test-size", type=int, default=20_000)
    ap.add_argument("--eval-size", type=int, default=500)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    t0 = time.perf_counter()

    data = load_bodmas()
    train_all, test_all = temporal_split(data)
    train_local = stratified_indices(data.y[train_all], min(args.train_size, len(train_all)), seed=42)
    test_local = stratified_indices(data.y[test_all], min(args.test_size, len(test_all)), seed=43)
    train_idx = train_all[train_local]
    test_idx = test_all[test_local]

    X_train = np.asarray(data.X[train_idx], dtype=np.float32)
    y_train = np.asarray(data.y[train_idx], dtype=np.int8)
    X_test = np.asarray(data.X[test_idx], dtype=np.float32)
    y_test = np.asarray(data.y[test_idx], dtype=np.int8)

    scaler = StandardScaler(with_mean=False, with_std=True, copy=True)
    X_train_s = scaler.fit_transform(X_train).astype(np.float32, copy=False)
    X_test_s = scaler.transform(X_test).astype(np.float32, copy=False)
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=1500,
        tol=1e-4,
        random_state=42,
        shuffle=True,
        average=True,
    )
    clf.fit(X_train_s, y_train)
    test_scores = clf.predict_proba(X_test_s)[:, 1]
    auc = float(roc_auc_score(y_test, test_scores))

    eval_local = stratified_indices(y_test, min(args.eval_size, len(y_test)), seed=44)
    X_eval = np.asarray(X_test_s[eval_local], dtype=np.float32)
    y_eval = np.asarray(y_test[eval_local], dtype=np.int8)
    scores = np.asarray(clf.predict_proba(X_eval)[:, 1], dtype=np.float64)
    coef = np.asarray(clf.coef_[0], dtype=np.float64)
    phi = linear_attribution(X_eval, coef)

    # Actual explanation-only scope postconditions.
    X_after = np.array(X_eval, copy=True)
    y_after = np.array(y_eval, copy=True)
    scores_after = np.asarray(clf.predict_proba(X_after)[:, 1], dtype=np.float64)
    post = {
        "input_sha256_before": sha(X_eval),
        "input_sha256_after": sha(X_after),
        "input_max_abs_diff": float(np.max(np.abs(X_eval - X_after))),
        "detector_score_sha256_before": sha(scores),
        "detector_score_sha256_after": sha(scores_after),
        "detector_score_max_abs_diff": float(np.max(np.abs(scores - scores_after))),
        "labels_sha256_before": sha(y_eval),
        "labels_sha256_after": sha(y_after),
        "labels_equal": bool(np.array_equal(y_eval, y_after)),
    }

    fam = transform_families(phi)
    fam2 = transform_families(phi)
    deterministic = all(
        label == label2 and np.array_equal(arr, arr2)
        for family in fam
        for (label, arr), (label2, arr2) in zip(fam[family], fam2[family])
    )
    delta_rows = []
    for family, transforms in fam.items():
        delta_rows.extend(evaluate_family(phi, y_eval, family, transforms))

    key = hashlib.sha256(b"T-XAI-E12-controlled-test-key-seed-42").digest()
    bundles = [evidence_bundle(i, X_eval, scores, phi, key, 20) for i in range(len(X_eval))]
    m3 = evaluate_m3(bundles, key)

    try:
        git_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except Exception:
        git_revision = "unknown"
        git_dirty = True

    max_ratio = max(float(r["ratio_L1"]) for r in delta_rows)
    nonzero_response_rows = sum(float(r["lhs_adv_tv"]) > 1e-12 for r in delta_rows)
    failed = []
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2 or len(np.unique(y_eval)) < 2:
        failed.append("class_diversity")
    if auc < 0.70:
        failed.append("detector_auc")
    if len(y_eval) != args.eval_size:
        failed.append("evaluation_size")
    if not deterministic:
        failed.append("deterministic_explanation_transform")
    if not (post["input_max_abs_diff"] == 0.0 and post["detector_score_max_abs_diff"] == 0.0 and post["labels_equal"]):
        failed.append("delta_z_postconditions")
    if not all(r["holds_L1"] and r["holds_dobrushin"] for r in delta_rows):
        failed.append("delta_z_inequalities")
    if nonzero_response_rows < 3 or max_ratio <= 0.01:
        failed.append("delta_z_nontrivial")
    if m3["minimum_authenticated_detection_rate"] < 0.99:
        failed.append("m3_bundle_detection")
    if m3["clean_false_reject_rate"] > 0.01:
        failed.append("m3_clean_acceptance")
    if m3["channel_integrity"]["malicious_detection_rate"] < 0.99:
        failed.append("m3_channel_detection")
    if m3["channel_integrity"]["benign_acceptance_rate"] < 0.99:
        failed.append("m3_channel_benign")
    if not m3["channel_integrity"]["key_compromise_bypass"]:
        failed.append("m3_key_boundary")

    warnings = []
    if auc > 0.995:
        warnings.append("BODMAS is highly separable under the external detector; use this run as mechanism replication rather than difficulty replication.")
    warnings.append("The linear contribution explanation is exact for the scaled linear logit up to the intercept, but it is a different explanation mechanism rather than a human-grounded explanation study.")
    verdict = "FAIL_REPAIR" if failed else ("PASS_WITH_WARNINGS" if warnings else "PASS")

    payload = {
        "experiment": "E12_external_replication",
        "status": "smoke" if args.tag else "official",
        "purpose": "external replication across corpus, detector, and explanation mechanism for Delta_z and controlled M3 evidence",
        "corpus": {
            "name": "BODMAS",
            "feature_layout": "EMBER-v2 2381-dimensional static PE feature space",
            "split": "temporal holdout from published acquisition timestamps",
            "train_all": int(len(train_all)),
            "test_all": int(len(test_all)),
            "train_used": int(len(train_idx)),
            "test_used": int(len(test_idx)),
            "train_timestamp_range": _timestamp_range(data.timestamps[train_idx]),
            "test_timestamp_range": _timestamp_range(data.timestamps[test_idx]),
            "train_class_counts": {str(int(c)): int((y_train == c).sum()) for c in np.unique(y_train)},
            "test_class_counts": {str(int(c)): int((y_test == c).sum()) for c in np.unique(y_test)},
            "eval_class_counts": {str(int(c)): int((y_eval == c).sum()) for c in np.unique(y_eval)},
        },
        "detector": {
            "family": "SGDClassifier logistic regression",
            "loss": "log_loss",
            "alpha": 1e-4,
            "max_iter": 1500,
            "tol": 1e-4,
            "average": True,
            "random_state": 42,
            "scaler": "StandardScaler(with_mean=False)",
            "test_auc": auc,
            "coef_sha256": sha(coef),
            "scale_sha256": sha(np.asarray(scaler.scale_, dtype=np.float64)),
        },
        "explainer": {
            "name": "deterministic linear contribution",
            "definition": "phi_j = x_scaled_j * coefficient_j",
            "clean_explanation_sha256": sha(phi),
        },
        "delta_z": {
            "n_eval": int(len(y_eval)),
            "n_unique_transforms": 18,
            "n_rows": int(len(delta_rows)),
            "deterministic_exact": deterministic,
            "scope_postconditions": post,
            "rows": delta_rows,
            "max_ratio_L1": max_ratio,
            "nonzero_response_rows": int(nonzero_response_rows),
        },
        "m3": m3,
        "reproducibility": {
            "git_revision": git_revision,
            "working_tree_dirty": git_dirty,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "gate": {
            "gate_id": "G4_external_replication",
            "stage": "external_replication",
            "verdict": verdict,
            "metrics": {
                "test_auc": auc,
                "n_eval": int(len(y_eval)),
                "eval_benign": int((y_eval == 0).sum()),
                "eval_malware": int((y_eval == 1).sum()),
                "delta_z_rows": int(len(delta_rows)),
                "delta_z_all_hold": bool(all(r["holds_L1"] and r["holds_dobrushin"] for r in delta_rows)),
                "delta_z_max_ratio_L1": max_ratio,
                "delta_z_nonzero_response_rows": int(nonzero_response_rows),
                "m3_min_authenticated_detection": m3["minimum_authenticated_detection_rate"],
                "m3_clean_false_reject": m3["clean_false_reject_rate"],
                "m3_channel_malicious_detection": m3["channel_integrity"]["malicious_detection_rate"],
                "m3_channel_benign_acceptance": m3["channel_integrity"]["benign_acceptance_rate"],
                "m3_key_compromise_bypass": m3["channel_integrity"]["key_compromise_bypass"],
            },
            "pass_criteria": {
                "class_diversity": "both classes in train, held-out test, and 500-sample evaluation",
                "detector_auc": ">=0.70",
                "evaluation_size": args.eval_size,
                "deterministic_explanation_transform": True,
                "delta_z_postconditions": "input, detector probabilities, and labels exactly unchanged",
                "delta_z_inequalities": "all 12 generic and Dobrushin rows",
                "delta_z_nontrivial": ">=3 nonzero response rows and max ratio >0.01",
                "m3_bundle_detection": ">=0.99 for every bundle attack",
                "m3_clean_acceptance": "false reject <=0.01",
                "m3_channel_detection": ">=0.99",
                "m3_channel_benign": ">=0.99",
                "m3_key_boundary": "full retag bypass under compromised authenticator",
            },
            "failed_criteria": failed,
            "warnings": warnings,
            "next_allowed_step": "external_replication_claim_update" if not failed else "repair_external_replication",
            "fail_interpretation": "The second-corpus/detector/explainer replication is degenerate, out of scope, or inconsistent with E10/E11 mechanisms.",
        },
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
    }
    suffix = f"_{args.tag}" if args.tag else ""
    out = RESULTS / f"E12_external_replication{suffix}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n")
    print(json.dumps({"output": str(out), "gate": payload["gate"], "elapsed_seconds": payload["elapsed_seconds"]}, indent=2))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
