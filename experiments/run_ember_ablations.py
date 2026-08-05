"""Ablations A2 and A7 on EMBER-2018.

`run_scoring_layer.py` runs both against BODMAS, which it loads directly. Section
VII now reports EMBER-2018, and an ablation subsection measured on the other
corpus would say nothing about the numbers it sits beside. This driver runs the
same two ablations through the same `run_explainer_block`, differing only in the
substrate builder.

Two deliberate departures from the main run, both to bound cost:

* the seed ablation refits the detector once per seed, and a fit on EMBER-2018
  costs about six minutes, so the reduced grid of `run_scoring_layer` is kept --
  one perturbation family, one budget;
* the detector ablation tries `xgboost` and `randomforest`, and a random forest
  on 105k x 2381 is slow. A detector that fails to build is recorded as an error
  row rather than dropped, so a missing arm cannot be mistaken for one that
  agreed.

Both write the perturbation budget into the result, which the manuscript's
earlier seed figure omitted.

Usage:
    python3 run_ember_ablations.py                 # both
    python3 run_ember_ablations.py --stage seeds
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_scoring_layer import environment, run_explainer_block, save  # noqa: E402
from txai_exp.config import ExperimentConfig  # noqa: E402
from txai_exp.ember.substrate import build_ember_substrate  # noqa: E402

logger = logging.getLogger("ablate_ember")


def seed_ablation(cfg: ExperimentConfig) -> dict:
    """A7 on EMBER: how much of the headline moves when only the seed does."""
    reduced = replace(cfg, n_perturb=10)
    rows = []
    for seed in range(cfg.seed, cfg.seed + cfg.n_seeds):
        sub, provenance = build_ember_substrate(reduced, "histgb", seed,
                                                reduced.n_robust)
        block, _, _ = run_explainer_block(
            None, reduced, sub, "treeshap",
            families=("append_bytes",), budgets=(reduced.perturb_strength,),
            n_perturb=10, n_explain=reduced.n_robust,
            n_robust=reduced.n_robust,
            note="REDUCED: one family, one budget, 10 perturbations")
        rows.append({"seed": seed, "auc": sub.auc,
                     "alert_malware_fraction":
                         float((sub.y_alerts == 1).mean()),
                     "train_size": provenance["train_size"], **block})
        logger.info("seed %d: AUC %.6f", seed, sub.auc)
        save("ablation_seeds_ember", _wrap(reduced, rows))
    return _wrap(reduced, rows)


def _wrap(cfg: ExperimentConfig, rows: list) -> dict:
    """Rows plus the perturbation budget they were measured at.

    The manuscript quoted a seed spread for the robustness score without saying
    which budget it was measured at, and the score moves by more across budgets
    than across seeds -- so the number was unlocatable. Recording the budget
    beside the rows makes that impossible to repeat.
    """
    return {"corpus": "EMBER-2018 feature version 2",
            "perturbation_family": "append_bytes",
            "perturbation_budget": cfg.perturb_strength,
            "n_perturb": cfg.n_perturb,
            "n_alerts": cfg.n_robust,
            "rows": rows}


def detector_ablation(cfg: ExperimentConfig) -> dict:
    """A2 on EMBER: does anything depend on which tree ensemble is used?"""
    reduced = replace(cfg, n_perturb=5)
    rows = {}
    for detector in ("xgboost", "randomforest"):
        t0 = time.perf_counter()
        try:
            sub, _ = build_ember_substrate(reduced, detector, reduced.seed,
                                           reduced.n_robust)
        except Exception as exc:                # noqa: BLE001
            logger.error("detector %s unavailable: %s", detector, exc)
            rows[detector] = {"error": str(exc)}
            save("ablation_detector_ember", rows)
            continue
        block, _, _ = run_explainer_block(
            None, reduced, sub, "treeshap",
            families=("append_bytes", "generic"),
            budgets=(reduced.perturb_strength,), n_perturb=5,
            n_explain=reduced.n_robust, n_robust=reduced.n_robust,
            note="REDUCED: two families, one budget, 5 perturbations")
        rows[detector] = {"auc": sub.auc, "alert_rate": sub.alert_rate,
                          "fit_seconds": sub.fit_seconds,
                          "wall_seconds": round(time.perf_counter() - t0, 1),
                          **block}
        logger.info("detector %s: AUC %.6f", detector, sub.auc)
        save("ablation_detector_ember", rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["all", "seeds", "detectors"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = ExperimentConfig()
    out = {"environment": environment(), "corpus": "ember"}
    t0 = time.perf_counter()

    if args.stage in ("all", "seeds"):
        out["ablation_seed"] = seed_ablation(cfg)
    if args.stage in ("all", "detectors"):
        out["ablation_detector"] = detector_ablation(cfg)

    out["total_seconds"] = round(time.perf_counter() - t0, 1)
    save("ablations_ember", out)
    logger.info("ablations on EMBER complete in %.1f min",
                out["total_seconds"] / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
