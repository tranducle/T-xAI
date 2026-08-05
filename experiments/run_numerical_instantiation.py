"""Section VII end to end, on either corpus, in one process.

The original Section VII was measured on BODMAS, where the detector reaches
AUC 0.99994 and every sampled alert is malware. Two consequences the review
pressed, and both are corpus properties rather than framework properties:

* the latent world H is constant, so the joint law over (world, action) equals
  its action marginal and the release game cannot exhibit the distinction
  Propositions `marginal` and `advbound` and Theorem `bridge` are about;
* eight of nine EMBER feature groups separate the corpus on their own, so every
  faithfulness, calibration and robustness number describes an explanation
  problem with almost nothing in it.

This driver takes `--corpus` and runs the whole section on either substrate, so
the two are measured by identical code and are comparable line for line. It also
runs the four additions: the D/F sensitivity sweep, the actionability condition,
the sufficiency-threshold sweep, and the positive controls.

Everything shares one substrate and one set of attributions. Fitting the
detector is the expensive step (~200 s on EMBER-2018), and running the stages in
separate processes would pay it once per stage.

Usage:
    python3 run_numerical_instantiation.py --corpus ember
    python3 run_numerical_instantiation.py --corpus bodmas --with-lime
    python3 run_numerical_instantiation.py --corpus ember --n-explain 50 --n-robust 50 \
        --train-size 8000 --test-size 5000 --tag smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_release_game import run_e8  # noqa: E402
from run_scoring_layer import RESULTS_DIR, environment, run_explainer_block, save  # noqa: E402
from txai_exp import supplementary, pipeline  # noqa: E402
from txai_exp.config import ExperimentConfig  # noqa: E402

logger = logging.getLogger("section7")

EXPLAINERS = ("treeshap", "random", "constant")


def build(corpus: str, cfg: ExperimentConfig, args) -> tuple:
    """(substrate, provenance) for the requested corpus."""
    if corpus == "bodmas":
        from txai_exp.data import load_bodmas
        data = load_bodmas()
        sub = pipeline.build_substrate(data, cfg, "histgb", cfg.seed,
                                       cfg.n_explain)
        return sub, {"corpus": "BODMAS", "split": "temporal, by timestamp",
                     "train_size": sub.train_size, "test_size": sub.test_size,
                     "auc": sub.auc, "alert_rate": sub.alert_rate,
                     "alert_malware_fraction":
                         float((sub.y_alerts == 1).mean()),
                     "alert_benign_count": int((sub.y_alerts == 0).sum())}

    from txai_exp.ember.substrate import build_ember_substrate
    kwargs = {}
    if args.train_size:
        kwargs["train_size"] = args.train_size
    if args.test_size:
        kwargs["test_size"] = args.test_size
    return build_ember_substrate(cfg, "histgb", cfg.seed, cfg.n_explain,
                                 **kwargs)


def separability(corpus: str, sub) -> dict:
    """The E1 record: substrate stats plus per-group separability.

    On EMBER the per-group AUCs are E9's rather than recomputed. E9 fitted nine
    single-group detectors on the same corpus, the same seed and the same
    training size this substrate uses, so refitting them would spend about
    fifteen minutes to reproduce numbers already on disk -- and if the two ever
    disagreed, that disagreement would be the finding, so the source is
    recorded rather than the value copied silently.
    """
    record = {
        "detector": sub.detector_name, "seed": sub.seed,
        "train_size": sub.train_size, "test_size": sub.test_size,
        "auc": sub.auc, "alert_rate": sub.alert_rate,
        "fit_seconds": sub.fit_seconds,
        "n_alerts_explained": int(len(sub.X_alerts)),
        "alert_malware_fraction": float((sub.y_alerts == 1).mean()),
        "alert_benign_count": int((sub.y_alerts == 0).sum()),
    }
    if corpus == "ember":
        path = RESULTS_DIR / "E9_ember2018.json"
        if path.exists():
            e9 = json.loads(path.read_text())
            record["auc_by_feature_group_alone"] = e9["auc_by_feature_group_alone"]
            record["auc_by_feature_group_source"] = (
                f"E9 ({path.name}), train_size={e9['train_size']}, "
                f"seed={e9['seed']}")
            record["groups_at_or_above_tau"] = e9["groups_at_or_above_tau"]
            record["separability_tau"] = e9["separability_tau"]
        else:
            record["auc_by_feature_group_alone"] = None
            record["auc_by_feature_group_source"] = "absent: E9 has not been run"
    return record


def run_blocks(cfg: ExperimentConfig, sub, use_lime: bool) -> tuple:
    """The per-explainer stages, keeping the arrays the add-ons need."""
    blocks, attributions, arrays = {}, {}, {}
    for name in EXPLAINERS:
        block, phi, arr = run_explainer_block(
            None, cfg, sub, name,
            families=cfg.perturbations, budgets=cfg.perturb_budget_grid,
            n_perturb=cfg.n_perturb, n_explain=cfg.n_explain,
            n_robust=cfg.n_robust, note="full grid")
        blocks[name], attributions[name], arrays[name] = block, phi, arr
        logger.info("block %s done in %.1fs", name, block["seconds"])

    if use_lime:
        block, phi, arr = run_explainer_block(
            None, cfg, sub, "lime",
            families=("append_bytes",), budgets=(cfg.perturb_strength,),
            n_perturb=5, n_explain=cfg.n_explain_lime,
            n_robust=cfg.n_robust_lime,
            note=("REDUCED: one perturbation family, one budget, 5 "
                  "perturbations. LIME is ~540x slower per instance than "
                  "TreeSHAP; its numbers are not directly comparable and must "
                  "not share a column with TreeSHAP's without this note."))
        blocks["lime"], attributions["lime"], arrays["lime"] = block, phi, arr
    return blocks, attributions, arrays


def run_addons(cfg: ExperimentConfig, sub, attributions: Dict[str, np.ndarray],
               arrays: Dict[str, dict], blocks: Dict[str, dict]) -> dict:
    """The four measurements added in response to the round-3 review."""
    # Not trimmed to a common length. LIME runs the reduced grid its cost
    # forces, so a shared prefix would report TreeSHAP's and both controls'
    # sweeps on LIME's alert count while labelling them as the full run. The
    # add-ons align F, B and D per explainer instead and record the count each
    # one used; the only cross-map difference taken (constant minus TreeSHAP)
    # is between two arms that both ran the full grid.
    B_by_explainer = {k: arrays[k]["B_default"] for k in attributions}

    # The perturbed attributions at the default family and budget, which the
    # actionability stability factor needs. Absent for an explainer whose
    # reduced grid did not include that cell, in which case stability is 1.0
    # and A is an upper estimate -- recorded per explainer, not assumed.
    perturbed = {}
    for name in attributions:
        cache = arrays[name]["cache"]
        key = (cfg.perturbations[0], cfg.perturb_strength)
        if key in cache:
            perturbed[name] = cache[key]

    rows = blocks["treeshap"]["E5_bridge"]["rows"]
    out = {
        "sensitivity": supplementary.sensitivity_sweep(sub, cfg, attributions,
                                                B_by_explainer),
        "actionability": supplementary.actionability_block(sub, cfg, attributions,
                                                    B_by_explainer, perturbed),
        "calibration_sweep": supplementary.calibration_block(sub, cfg, attributions),
        "controls": supplementary.controls_block(sub, cfg,
                                          arrays["treeshap"]["phi_robust"],
                                          bridge=_default_bridge(cfg, rows)),
    }
    # The default configuration is not the informative one: the falsification
    # margin is smallest where the bound is tightest, and that is the number
    # that says how much slack Theorem 1 really has on this substrate.
    out["controls"]["bridge_margin_tightest"] = _tightest_margin(rows)
    return out


def _default_bridge(cfg: ExperimentConfig, rows: list) -> Optional[dict]:
    """The E5 row at the default family, budget and kernel temperature.

    Selected by name, not by position: taking `rows[0]` would make the reported
    falsification margin depend on dict iteration order, and a margin quoted for
    an unnamed configuration cannot be checked.
    """
    for row in rows:
        if (row["family"] == cfg.perturbations[0]
                and row["strength"] == cfg.perturb_strength
                and row["temperature"] == 1.0):
            return row
    logger.warning("no E5 row at the default configuration; the bridge control "
                   "is reported as unavailable rather than on a substitute")
    return None


def _tightest_margin(rows: list) -> Optional[dict]:
    """The falsification margin at the configuration where the bound is tightest."""
    from txai_exp.controls import bridge_falsification_margin

    usable = [r for r in rows if np.isfinite(r["tightness_ratio"])
              and r["tightness_ratio"] > 0]
    if not usable:
        return None
    row = max(usable, key=lambda r: r["tightness_ratio"])
    margin = bridge_falsification_margin(row)
    margin["configuration"] = {k: row[k] for k in
                               ("family", "strength", "temperature",
                                "tightness_ratio")}
    return margin


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="ember", choices=["ember", "bodmas"])
    parser.add_argument("--with-lime", action="store_true")
    parser.add_argument("--n-explain", type=int, default=None)
    parser.add_argument("--n-robust", type=int, default=None)
    parser.add_argument("--n-perturb", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--tag", default=None,
                        help="suffix for the output files; use for smoke runs "
                             "so they cannot overwrite a full run's results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    t0 = time.perf_counter()

    cfg = ExperimentConfig()
    overrides = {k: v for k, v in
                 (("n_explain", args.n_explain), ("n_robust", args.n_robust),
                  ("n_perturb", args.n_perturb)) if v is not None}
    if overrides:
        cfg = replace(cfg, **overrides)
        logger.warning("configuration overridden: %s -- this is a reduced run "
                       "and its numbers are not the section's", overrides)

    sub, provenance = build(args.corpus, cfg, args)
    logger.info("substrate: AUC=%.5f, %d alerts, %.4f malware",
                sub.auc, len(sub.X_alerts),
                float((sub.y_alerts == 1).mean()))

    blocks, attributions, arrays = run_blocks(cfg, sub, args.with_lime)
    n_min = min(len(p) for p in attributions.values())
    trimmed = {k: v[:n_min] for k, v in attributions.items()}

    out = {
        "environment": environment(),
        "corpus": args.corpus,
        "provenance": provenance,
        "config": dict(vars(cfg)),
        "reduced_run": bool(overrides) or bool(args.train_size
                                               or args.test_size),
        "E1_substrate": separability(args.corpus, sub),
        "E2_faithfulness": pipeline.e2_faithfulness(sub, cfg, trimmed),
        "E3_calibration": pipeline.e3_calibration(sub, cfg, trimmed),
        "E6_disclosure": pipeline.e6_disclosure(sub, cfg, trimmed),
        "blocks": blocks,
        "cross_explainer_n": n_min,
    }
    logger.info("E1-E7 done in %.1f min", (time.perf_counter() - t0) / 60)

    out["E8_release_game"] = run_e8(sub, cfg)
    logger.info("E8 done; world mass %s", out["E8_release_game"]["world_mass"])

    # Checkpoint before the add-ons. The detector fit and the explanations are
    # the expensive part of this run, and losing them to a fault in a stage
    # added later would mean paying for them twice.
    suffix = f"_{args.tag}" if args.tag else ""
    save(f"section7_{args.corpus}{suffix}_partial", out)

    # The key stays `addons`: it is the schema of the results files already
    # published under `results/`, and `docs/ROUND3_EMBER.md` cites paths into
    # it. Only the module that fills it was renamed.
    out["addons"] = run_addons(cfg, sub, attributions, arrays, blocks)
    out["total_seconds"] = round(time.perf_counter() - t0, 1)

    save(f"section7_{args.corpus}{suffix}", out)
    logger.info("Section VII on %s complete in %.1f min",
                args.corpus, out["total_seconds"] / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
