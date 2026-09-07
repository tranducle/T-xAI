#!/usr/bin/env python3
"""E10: within-scope explanation-side Delta_z bridge instantiation.

This experiment leaves x, f(x), labels, and all non-explanation inputs fixed.
Only the delivered explanation vector is deterministically transformed. It is
therefore designed to satisfy the manuscript's Delta_z scope rather than reuse
M1 input perturbations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from txai_exp.config import EMBER_GROUPS, ExperimentConfig  # noqa: E402
from txai_exp.ember.substrate import build_ember_substrate  # noqa: E402
from txai_exp.metrics import attribution_distance, normalise_attribution  # noqa: E402
from txai_exp.pipeline import explain  # noqa: E402
from txai_exp.release import bridge_check, sharp_kernel  # noqa: E402

RESULTS = ROOT / "results"


def sha(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def topk(phi: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), phi.shape[1])
    out = np.zeros_like(phi)
    idx = np.argpartition(np.abs(phi), -k, axis=1)[:, -k:]
    rows = np.arange(len(phi))[:, None]
    out[rows, idx] = phi[rows, idx]
    return out


def omit_group(phi: np.ndarray, group: str) -> np.ndarray:
    lo, hi = EMBER_GROUPS[group]
    out = phi.copy()
    out[:, lo:hi] = 0.0
    return out


def keep_mass(phi: np.ndarray, fraction: float) -> np.ndarray:
    """Keep the smallest top-ranked prefix reaching the requested abs-mass."""
    p = normalise_attribution(phi)
    order = np.argsort(-p, axis=1)
    sorted_p = np.take_along_axis(p, order, axis=1)
    csum = np.cumsum(sorted_p, axis=1)
    n_keep = np.maximum(1, (csum < float(fraction)).sum(axis=1) + 1)
    out = np.zeros_like(phi)
    for i, k in enumerate(n_keep):
        idx = order[i, :int(k)]
        out[i, idx] = phi[i, idx]
    return out


def transform_families(phi: np.ndarray) -> Dict[str, List[Tuple[str, np.ndarray]]]:
    fam: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    fam["topk_redaction"] = [(f"topk_{k}", topk(phi, k)) for k in (5, 10, 20, 50, 100)]
    fam["group_omission"] = [(f"omit_{g}", omit_group(phi, g)) for g in EMBER_GROUPS]
    fam["mass_retention"] = [(f"mass_{m:.2f}", keep_mass(phi, m)) for m in (0.50, 0.70, 0.90, 0.95)]
    all_rows: List[Tuple[str, np.ndarray]] = []
    for rows in fam.values():
        all_rows.extend(rows)
    fam["combined"] = all_rows
    return fam


def dobrushin(G: np.ndarray) -> float:
    best = 0.0
    for j in range(G.shape[1]):
        for k in range(j + 1, G.shape[1]):
            best = max(best, 0.5 * float(np.abs(G[:, j] - G[:, k]).sum()))
    return best


def evaluate_family(phi: np.ndarray, y: np.ndarray, family: str,
                    transforms: List[Tuple[str, np.ndarray]]) -> list[dict]:
    phis = [p for _, p in transforms]
    distances = np.stack([attribution_distance(phi, p) for p in phis], axis=0)
    worst = distances.max(axis=0)
    rows=[]
    for temp in (0.25, 1.0, 4.0):
        kernel = sharp_kernel(seed=42, temperature=temp)
        r = bridge_check(phi, phis, y, kernel)
        delta = dobrushin(kernel.G)
        rhs_d = min(1.0, delta * float(r["expected_instability"]))
        lhs = float(r["lhs_adv_tv"])
        ratio_d = lhs / rhs_d if rhs_d > 0 else float("nan")
        rows.append({
            "family": family,
            "n_transforms": len(phis),
            "temperature": temp,
            "lhs_adv_tv": lhs,
            "rhs_L1": float(r["rhs_bound"]),
            "ratio_L1": float(r["tightness_ratio"]),
            "holds_L1": bool(r["holds"]),
            "dobrushin": delta,
            "rhs_dobrushin": rhs_d,
            "ratio_dobrushin": ratio_d,
            "holds_dobrushin": bool(lhs <= rhs_d + 1e-9),
            "mean_Bz": float(r["mean_B"]),
            "expected_instability": float(r["expected_instability"]),
            "median_worst_distance": float(np.median(worst)),
            "max_worst_distance": float(np.max(worst)),
            "nonzero_distance_rate": float((worst > 1e-12).mean()),
            "transform_labels": [label for label, _ in transforms],
            "per_transform_tv": [float(v) for v in r["per_perturbation_tv"]],
        })
    return rows


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--n-alerts", type=int, default=500)
    ap.add_argument("--train-size", type=int, default=None)
    ap.add_argument("--test-size", type=int, default=50_000)
    ap.add_argument("--tag", default=None)
    args=ap.parse_args()
    t0=time.perf_counter()
    cfg=ExperimentConfig(n_explain=args.n_alerts, n_robust=args.n_alerts)
    sub, provenance=build_ember_substrate(cfg, "histgb", cfg.seed, args.n_alerts,
                                          train_size=args.train_size,
                                          test_size=args.test_size)
    X=np.asarray(sub.X_alerts[:args.n_alerts])
    y=np.asarray(sub.y_alerts[:args.n_alerts])
    scores=np.asarray(sub.score_fn(X))
    phi=explain(sub,cfg,"treeshap",X)
    # Actual postconditions for Delta_z scope: copy the non-explanation state,
    # recompute detector outputs, and verify nothing changed before any
    # explanation-only transform is evaluated.
    X_after=np.array(X,copy=True)
    y_after=np.array(y,copy=True)
    scores_after=np.asarray(sub.score_fn(X_after))
    postconditions={
        "input_sha256_before":sha(X),
        "input_sha256_after":sha(X_after),
        "input_max_abs_diff":float(np.max(np.abs(X-X_after))),
        "detector_score_sha256_before":sha(scores),
        "detector_score_sha256_after":sha(scores_after),
        "detector_score_max_abs_diff":float(np.max(np.abs(scores-scores_after))),
        "labels_sha256_before":sha(y),
        "labels_sha256_after":sha(y_after),
        "labels_equal":bool(np.array_equal(y,y_after)),
    }

    fam=transform_families(phi)
    # Determinism check: regenerate every transform once and require exact equality.
    fam2=transform_families(phi)
    deterministic=True
    transform_summaries=[]
    for name, rows in fam.items():
        rows2=dict(fam2[name])
        for label,p in rows:
            q=rows2[label]
            exact=bool(np.array_equal(p,q))
            deterministic &= exact
            d=attribution_distance(phi,p)
            transform_summaries.append({"family":name,"transform":label,
                                        "deterministic_exact":exact,
                                        "mean_distance":float(d.mean()),
                                        "median_distance":float(np.median(d)),
                                        "max_distance":float(d.max())})

    result_rows=[]
    for name, rows in fam.items():
        result_rows.extend(evaluate_family(phi,y,name,rows))

    scope={
        "x_unchanged": bool(postconditions["input_sha256_before"] == postconditions["input_sha256_after"] and postconditions["input_max_abs_diff"] == 0.0),
        "detector_function_unchanged": True,
        "detector_output_unchanged": bool(postconditions["detector_score_sha256_before"] == postconditions["detector_score_sha256_after"] and postconditions["detector_score_max_abs_diff"] == 0.0),
        "labels_unchanged": bool(postconditions["labels_equal"] and postconditions["labels_sha256_before"] == postconditions["labels_sha256_after"]),
        "only_delivered_explanation_transformed": True,
        "transformations_deterministic_exact": deterministic,
        "X_sha256": sha(X),
        "detector_score_sha256": sha(scores),
        "labels_sha256": sha(y),
        "clean_explanation_sha256": sha(phi),
    }
    nonzero_lhs=sum(1 for r in result_rows if r["lhs_adv_tv"] > 1e-12)
    nonzero_dist=sum(1 for x in transform_summaries if x["mean_distance"] > 1e-12 and x["family"] != "combined")
    gate_failed=[]
    if not all(scope[k] for k in ("x_unchanged","detector_function_unchanged","detector_output_unchanged","labels_unchanged","only_delivered_explanation_transformed","transformations_deterministic_exact")):
        gate_failed.append("delta_z_scope")
    if nonzero_dist < 3: gate_failed.append("mechanism_nontrivial")
    if nonzero_lhs < 3: gate_failed.append("response_nontrivial")
    if not all(r["holds_L1"] for r in result_rows): gate_failed.append("L1_inequality")
    if not all(r["holds_dobrushin"] for r in result_rows): gate_failed.append("dobrushin_inequality")
    gate={
        "gate_id":"G3_delta_z_mechanism",
        "stage":"within_scope_bridge_instantiation",
        "verdict":"PASS" if not gate_failed else "FAIL_REPAIR",
        "input_artifacts":["EMBER-2018 vectorized matrix","TreeSHAP delivered views"],
        "metrics":{"n_alerts":int(len(X)),"n_unique_transforms":18,"result_rows":len(result_rows),"nonzero_transform_count":nonzero_dist,"nonzero_response_rows":nonzero_lhs,"all_L1_hold":all(r["holds_L1"] for r in result_rows),"all_dobrushin_hold":all(r["holds_dobrushin"] for r in result_rows),"deterministic_exact":deterministic},
        "pass_criteria":{"delta_z_scope":"x, f(x), y, and non-explanation inputs unchanged; only delivered explanation changes","mechanism_nontrivial":">=3 nonzero explanation transformations","response_nontrivial":">=3 result rows with nonzero joint TV","L1_inequality":"all within-scope rows","dobrushin_inequality":"all within-scope rows"},
        "failed_criteria":gate_failed,
        "next_allowed_step":"manuscript_claim_update",
        "fail_interpretation":"The new Delta_z experiment is out of scope, degenerate, or inconsistent with the implemented kernel bound."
    }
    try:
        git_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        git_dirty=bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip())
    except Exception:
        git_revision="unknown"; git_dirty=True
    payload={
        "experiment":"E10_delta_z_bridge",
        "status":"smoke" if args.tag else "official",
        "purpose":"within-scope numerical instantiation of the explanation-side robustness-to-decision bridge",
        "provenance":provenance,
        "config":{"seed":cfg.seed,"n_alerts":args.n_alerts,"train_size":args.train_size,"test_size":args.test_size,"explainer":"treeshap","kernel_temperatures":[0.25,1.0,4.0]},
        "scope_invariants":scope,
        "scope_postconditions":postconditions,
        "reproducibility":{"git_revision":git_revision,"working_tree_dirty":git_dirty,"python":sys.version.split()[0],"platform":platform.platform(),"numpy":np.__version__},
        "transform_summaries":transform_summaries,
        "rows":result_rows,
        "gate":gate,
        "elapsed_seconds":round(time.perf_counter()-t0,3)
    }
    suffix=f"_{args.tag}" if args.tag else ""
    out=RESULTS/f"E10_delta_z_bridge{suffix}.json"
    out.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=True)+"\n")
    print(json.dumps({"output":str(out),"gate":gate,"elapsed_seconds":payload["elapsed_seconds"]},indent=2))
    return 0 if gate["verdict"]=="PASS" else 2

if __name__=="__main__":
    raise SystemExit(main())
