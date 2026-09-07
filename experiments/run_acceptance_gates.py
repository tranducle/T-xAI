#!/usr/bin/env python3
"""Deterministic pre-run gates for the T-XAI acceptance-upgrade experiments."""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from txai_exp.ember.loader import load_ember2018  # noqa: E402

OUT = ROOT / "results" / "gates"
OUT.mkdir(parents=True, exist_ok=True)
MATRIX = Path(os.environ.get("TXAI_EMBER_MATRIX", ""))
if not str(MATRIX):
    raise SystemExit("TXAI_EMBER_MATRIX is required")


def dump(name: str, payload: dict) -> None:
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"{payload['gate_id']}: {payload['verdict']} -> {path}")


def g0(corpus) -> dict:
    required = ["manifest.json", "X.dat", "y.dat", "month.dat", "subset.dat", "sha256.dat", "avclass.dat", "STATUS.txt"]
    files = {name: {"exists": (MATRIX / name).exists(), "size_bytes": (MATRIX / name).stat().st_size if (MATRIX / name).exists() else 0} for name in required}
    manifest = json.loads((MATRIX / "manifest.json").read_text())
    chunks_done = len(list((MATRIX / "chunks").glob("*.done")))
    n_chunks = len(manifest["chunks"])
    counts = {str(v): int((corpus.y == v).sum()) for v in (-1, 0, 1)}
    failed = []
    if not all(v["exists"] and v["size_bytes"] > 0 for v in files.values()): failed.append("required_files")
    if int(manifest.get("n_rows", -1)) != 1_000_000: failed.append("n_rows")
    if int(manifest.get("n_features", -1)) != 2381: failed.append("n_features")
    if chunks_done != n_chunks: failed.append("all_chunks_complete")
    if counts["0"] == 0 or counts["1"] == 0: failed.append("label_diversity")
    return {
        "gate_id": "G0_data_provenance",
        "stage": "dataset_acquisition",
        "verdict": "PASS" if not failed else "FAIL_REPAIR",
        "input_artifacts": [f"external:EMBER-2018-vectorized/{x}" for x in required],
        "metrics": {"n_rows": int(manifest["n_rows"]), "n_features": int(manifest["n_features"]), "chunks_done": chunks_done, "n_chunks": n_chunks, "label_counts": counts, "bad_rows": int(len(corpus.bad_rows)), "files": files},
        "pass_criteria": {"required_files": "all present and non-empty", "n_rows": 1_000_000, "n_features": 2381, "all_chunks_complete": True, "label_diversity": "both benign and malware labels present"},
        "failed_criteria": failed,
        "next_allowed_step": "split_integrity",
        "fail_interpretation": "The vectorized EMBER corpus is incomplete, malformed, or label-collapsed."
    }


def g1(corpus) -> dict:
    train = corpus.select("train", labeled_only=True)
    test = corpus.select("test", labeled_only=True)
    train_hash = np.asarray(corpus.sha256[train])
    test_hash = np.asarray(corpus.sha256[test])
    overlap = np.intersect1d(train_hash, test_hash, assume_unique=False)
    train_labels = np.asarray(corpus.y[train])
    test_labels = np.asarray(corpus.y[test])
    train_month = np.asarray(corpus.month[train])
    test_month = np.asarray(corpus.month[test])
    metrics = {
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "train_benign": int((train_labels == 0).sum()), "train_malware": int((train_labels == 1).sum()),
        "test_benign": int((test_labels == 0).sum()), "test_malware": int((test_labels == 1).sum()),
        "sha_overlap_count": int(len(overlap)),
        "train_month_min": int(train_month[train_month >= 0].min()), "train_month_max": int(train_month[train_month >= 0].max()),
        "test_month_min": int(test_month[test_month >= 0].min()), "test_month_max": int(test_month[test_month >= 0].max()),
        "unknown_month_train": int((train_month < 0).sum()), "unknown_month_test": int((test_month < 0).sum()),
    }
    failed=[]
    if len(train) < 100_000 or len(test) < 100_000: failed.append("minimum_split_size")
    if metrics["train_benign"] == 0 or metrics["train_malware"] == 0 or metrics["test_benign"] == 0 or metrics["test_malware"] == 0: failed.append("label_diversity")
    if metrics["sha_overlap_count"] != 0: failed.append("sha_split_overlap")
    if metrics["train_month_max"] >= metrics["test_month_min"]: failed.append("temporal_order")
    return {
        "gate_id":"G1_preprocessing_integrity", "stage":"released_temporal_split",
        "verdict":"PASS" if not failed else "FAIL_REPAIR",
        "input_artifacts":["external:EMBER-2018-vectorized/sha256.dat", "external:EMBER-2018-vectorized/subset.dat", "external:EMBER-2018-vectorized/y.dat", "external:EMBER-2018-vectorized/month.dat"],
        "metrics":metrics,
        "pass_criteria":{"minimum_split_size":">=100000 labeled rows per split", "label_diversity":"both classes in train and test", "sha_split_overlap":0, "temporal_order":"max(train month) < min(test month)"},
        "failed_criteria":failed, "next_allowed_step":"baseline_sanity",
        "fail_interpretation":"The released split leaks records, collapses a class, or violates temporal ordering."
    }


def g2() -> dict:
    path = ROOT / "results" / "section7_ember.json"
    d=json.loads(path.read_text())
    e=d["E1_substrate"]
    auc=float(e["auc"]); alert=float(e["alert_rate"]); n=int(e["n_alerts_explained"])
    benign=int(e["alert_benign_count"]); malware=n-benign
    vals=[]
    nonfinite=[]
    def walk(x,path=""):
        if isinstance(x,dict):
            for k,v in x.items(): walk(v, f"{path}.{k}" if path else k)
        elif isinstance(x,list):
            for i,v in enumerate(x): walk(v, f"{path}[{i}]")
        elif isinstance(x,(int,float)) and not isinstance(x,bool):
            value=float(x); vals.append(value)
            if not math.isfinite(value): nonfinite.append(path)
    walk(d)
    allowed_nonfinite=[]
    unexpected_nonfinite=[]
    for path_name in nonfinite:
        if path_name.startswith("blocks.constant.E5_bridge.rows[") and path_name.endswith(".tightness_ratio"):
            allowed_nonfinite.append(path_name)
        else:
            unexpected_nonfinite.append(path_name)
    finite=not unexpected_nonfinite
    failed=[]
    if not (0.5 < auc < 1.0): failed.append("nondegenerate_auc")
    if not (0.01 < alert < 0.99): failed.append("noncollapsed_alert_rate")
    if benign <= 0 or malware <= 0: failed.append("alert_label_diversity")
    if not finite: failed.append("finite_claim_metrics")
    return {
        "gate_id":"G2_baseline_sanity", "stage":"baseline_evaluation",
        "verdict":"PASS" if not failed else "FAIL_REPAIR", "input_artifacts":["results/section7_ember.json"],
        "metrics":{"auc":auc,"alert_rate":alert,"n_alerts":n,"alert_benign":benign,"alert_malware":malware,"numeric_values_checked":len(vals),"unexpected_nonfinite":unexpected_nonfinite,"allowed_structural_nan_count":len(allowed_nonfinite),"allowed_structural_nan_reason":"constant-control bridge has LHS=RHS=0, so the ratio is 0/0 and intentionally undefined"},
        "pass_criteria":{"nondegenerate_auc":"0.5 < AUC < 1.0","noncollapsed_alert_rate":"0.01 < rate < 0.99","alert_label_diversity":"both benign and malware alerts present","finite_claim_metrics":"no non-finite values outside documented 0/0 constant-control tightness ratios"},
        "failed_criteria":failed,"next_allowed_step":"acceptance_upgrade_experiments",
        "fail_interpretation":"The baseline detector/result pipeline is degenerate or numerically invalid."
    }


def main() -> int:
    corpus = load_ember2018(MATRIX, verify=True)
    reports=[g0(corpus),g1(corpus),g2()]
    for r,n in zip(reports,["G0_data_provenance.json","G1_preprocessing_integrity.json","G2_baseline_sanity.json"]): dump(n,r)
    return 0 if all(r["verdict"]=="PASS" for r in reports) else 2

if __name__ == "__main__":
    raise SystemExit(main())
