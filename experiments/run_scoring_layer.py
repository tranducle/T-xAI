"""Driver for E1-E7 and ablations A1-A7.

Coverage is deliberately uneven and every reduction is recorded in the output
under `coverage_notes`, because a silently truncated grid reads as full coverage
once it reaches a results table. LIME costs ~0.65 s per instance against
TreeSHAP's ~0.0012 s, so a LIME grid matching TreeSHAP's would take about four
days on this machine; LIME therefore runs a reduced grid and the reduction is
reported next to its numbers.

Usage:
    python3 run_scoring_layer.py --stage all
    python3 run_scoring_layer.py --stage main --no-lime
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from txai import pipeline  # noqa: E402
from txai.config import ExperimentConfig  # noqa: E402
from txai.data import load_bodmas  # noqa: E402
from txai.metrics import faithfulness_F  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
logger = logging.getLogger("run")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"not JSON-serialisable: {type(obj)}")


def save(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    logger.info("wrote %s (%.1f KB)", path.name, path.stat().st_size / 1024)
    return path


def environment() -> dict:
    """Version record for the reproducibility appendix.

    Reads the installed distribution metadata rather than a module's
    ``__version__``: `lime` 0.2.0.1 defines no ``__version__`` at all, so the
    attribute probe recorded it as "absent" in the 2026-08-02 run even though
    the LIME block ran for 345 s in that same run. An environment record that
    denies a package that was used is worse than no record.
    """
    import importlib.metadata as md

    import sklearn
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "machine": platform.machine(), "numpy": np.__version__,
           "sklearn": sklearn.__version__}
    for dist in ("shap", "lime", "xgboost"):
        try:
            env[dist] = md.version(dist)
        except md.PackageNotFoundError:         # absence is data, not an error
            env[dist] = "absent"
    return env


# ---------------------------------------------------------------------------

def run_explainer_block(data, cfg, sub, explainer_name, *, families, budgets,
                        n_perturb, n_explain, n_robust, note) -> dict:
    """Everything that depends on one (detector, explainer) pair."""
    t0 = time.perf_counter()
    X_explain = sub.X_alerts[:n_explain]
    X_robust = sub.X_alerts[:n_robust]
    y_robust = sub.y_alerts[:n_robust]

    phi_explain = pipeline.explain(sub, cfg, explainer_name, X_explain)
    phi_robust = (phi_explain[:n_robust] if n_robust <= n_explain
                  else pipeline.explain(sub, cfg, explainer_name, X_robust))

    cache = pipeline.perturbed_attributions(
        sub, cfg, explainer_name, X_robust, families, budgets, n_perturb)

    F_robust = faithfulness_F(sub.score_fn, X_robust, phi_robust,
                              sub.reference, steps=cfg.deletion_steps)
    from txai.metrics import robustness_B
    B_default = robustness_B(
        phi_robust, cache[(families[0], cfg.perturb_strength)]
        if (families[0], cfg.perturb_strength) in cache
        else cache[next(iter(cache))])

    block = {
        "explainer": explainer_name,
        "coverage_notes": note,
        "grid": {"families": list(families), "budgets": list(budgets),
                 "n_perturb": n_perturb, "n_explain": n_explain,
                 "n_robust": n_robust},
        "E4_robustness": pipeline.e4_robustness(sub, cfg, explainer_name,
                                                phi_robust, cache),
        "E5_bridge": pipeline.e5_bridge(sub, cfg, explainer_name, phi_robust,
                                        y_robust, cache),
        "E7_admissibility": pipeline.e7_admissibility(
            sub, cfg, explainer_name, F_robust, B_default, phi_robust),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    return block, phi_explain


def run_main(data, cfg, use_lime: bool) -> dict:
    """E1-E7 on the primary substrate, with ablations A1 (explainer) and A3-A6."""
    sub = pipeline.build_substrate(data, cfg, "histgb", cfg.seed, cfg.n_explain)

    out = {"environment": environment(), "config": vars(cfg).copy()
           if hasattr(cfg, "__dict__") else cfg.__dict__.copy(),
           "E1_substrate": pipeline.e1_detector(data, cfg, sub)}
    save("E1_substrate", out["E1_substrate"])

    attributions = {}
    blocks = {}

    for name in ("treeshap", "random", "constant"):
        block, phi = run_explainer_block(
            data, cfg, sub, name,
            families=cfg.perturbations, budgets=cfg.perturb_budget_grid,
            n_perturb=cfg.n_perturb, n_explain=cfg.n_explain,
            n_robust=cfg.n_robust,
            note="full grid")
        blocks[name] = block
        attributions[name] = phi
        save(f"block_{name}", block)

    if use_lime:
        block, phi = run_explainer_block(
            data, cfg, sub, "lime",
            families=("append_bytes",), budgets=(cfg.perturb_strength,),
            n_perturb=5, n_explain=cfg.n_explain_lime,
            n_robust=cfg.n_robust_lime,
            note=("REDUCED: one perturbation family, one budget, 5 perturbations, "
                  "200 alerts explained and 100 scored for B. LIME is ~540x "
                  "slower per instance than TreeSHAP; the full grid would take "
                  "about four days on this machine. LIME numbers are therefore "
                  "not directly comparable to TreeSHAP's and must not be placed "
                  "in the same column without this note."))
        blocks["lime"] = block
        attributions["lime"] = phi
        save("block_lime", block)

    n_min = min(len(p) for p in attributions.values())
    trimmed = {k: v[:n_min] for k, v in attributions.items()}

    out["E2_faithfulness"] = pipeline.e2_faithfulness(sub, cfg, trimmed)
    out["E3_calibration"] = pipeline.e3_calibration(sub, cfg, trimmed)
    out["E6_disclosure"] = pipeline.e6_disclosure(sub, cfg, trimmed)
    out["blocks"] = blocks
    out["cross_explainer_n"] = n_min
    save("E2_faithfulness", out["E2_faithfulness"])
    save("E3_calibration", out["E3_calibration"])
    save("E6_disclosure", out["E6_disclosure"])
    return out


def run_detector_ablation(data, cfg) -> dict:
    """Ablation A2: does anything above depend on which tree ensemble is used?"""
    rows = {}
    reduced = replace(cfg, n_perturb=5)
    for detector in ("xgboost", "randomforest"):
        try:
            sub = pipeline.build_substrate(data, reduced, detector,
                                           reduced.seed, reduced.n_robust)
        except Exception as exc:                # noqa: BLE001
            logger.error("detector %s unavailable: %s", detector, exc)
            rows[detector] = {"error": str(exc)}
            continue
        block, _ = run_explainer_block(
            data, reduced, sub, "treeshap",
            families=("append_bytes", "generic"),
            budgets=(reduced.perturb_strength,), n_perturb=5,
            n_explain=reduced.n_robust, n_robust=reduced.n_robust,
            note="REDUCED: two families, one budget, 5 perturbations")
        rows[detector] = {"auc": sub.auc, "alert_rate": sub.alert_rate,
                          "fit_seconds": sub.fit_seconds, **block}
        save(f"ablation_detector_{detector}", rows[detector])
    return rows


def run_seed_ablation(data, cfg) -> dict:
    """Ablation A7: seed sensitivity of the headline quantities."""
    rows = []
    reduced = replace(cfg, n_perturb=10)
    for seed in range(cfg.seed, cfg.seed + cfg.n_seeds):
        sub = pipeline.build_substrate(data, reduced, "histgb", seed,
                                       reduced.n_robust)
        block, _ = run_explainer_block(
            data, reduced, sub, "treeshap",
            families=("append_bytes",), budgets=(reduced.perturb_strength,),
            n_perturb=10, n_explain=reduced.n_robust,
            n_robust=reduced.n_robust,
            note="REDUCED: one family, one budget, 10 perturbations")
        rows.append({"seed": seed, "auc": sub.auc, **block})
        save("ablation_seeds", {"rows": rows})
    return {"rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["all", "main", "detectors", "seeds"])
    parser.add_argument("--no-lime", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout)

    t0 = time.perf_counter()
    data = load_bodmas()
    logger.info("loaded in %.1fs", time.perf_counter() - t0)

    cfg = ExperimentConfig()
    results = {"environment": environment()}

    if args.stage in ("all", "main"):
        results["main"] = run_main(data, cfg, use_lime=not args.no_lime)
    if args.stage in ("all", "detectors"):
        results["ablation_detector"] = run_detector_ablation(data, cfg)
    if args.stage in ("all", "seeds"):
        results["ablation_seed"] = run_seed_ablation(data, cfg)

    results["total_seconds"] = round(time.perf_counter() - t0, 1)
    save(f"all_{args.stage}", results)
    logger.info("done in %.1f min", results["total_seconds"] / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
