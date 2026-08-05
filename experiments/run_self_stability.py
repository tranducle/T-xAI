"""E4b / ablation A8 -- self-stability: is the explanation map a *function* of x?

Eq. (6) of the manuscript (paper.tex:594) introduces $B_{\\Gamma,r}$ with an
explicit precondition: "For delivered views that are deterministic conditional
on $x$". Run 2 of the main programme shows the precondition is not free. With
the explainer instance no longer rebuilt per call, LIME at 2,000 samples scores
B = 0.595 under a 1 % input perturbation, against the random control's 0.575 --
i.e. almost none of the measured instability need come from the perturbation at
all.

This script separates the two sources by measuring B with an *identity*
perturbation set: same rows, same everything, K repeat explanations.

    B_self  = 1 - sup_k d(phi(x), phi_k(x))     with x unchanged
    B_pert  = the main programme's number        with x perturbed

B_self = 1 exactly means the map is deterministic and Eq. (6) applies as
written. B_self < 1 means the supremum in Eq. (6) is contaminated by the
explainer's own sampling noise, and any B reported for that explainer is a
statement about the sampling budget as much as about robustness.

The LIME sweep over `num_samples` then answers the reviewer's question -- is
LIME unstable, or was it merely under-sampled here?

Runs on either corpus. Section VII now reports EMBER-2018, and the determinism
precondition is a property of the explanation map rather than of the corpus --
but the numbers beside which it is quoted are EMBER's, so measuring it on BODMAS
would put two substrates in one paragraph.

Usage:
    python3 run_self_stability.py --corpus ember
    python3 run_self_stability.py --corpus bodmas
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_scoring_layer import environment, save  # noqa: E402
from txai_exp import pipeline  # noqa: E402
from txai_exp.config import ExperimentConfig  # noqa: E402
from txai_exp.data import load_bodmas  # noqa: E402
from txai_exp.metrics import robustness_B, summarise  # noqa: E402

logger = logging.getLogger("selfstab")

N_ALERTS = 100          # LIME-bound; TreeSHAP and the controls could take 500
N_REPEAT = 5
LIME_SAMPLE_GRID = (500, 2000, 8000)


def self_stability(sub, cfg, explainer_name: str, X: np.ndarray,
                   n_repeat: int = N_REPEAT) -> dict:
    """B under an identity perturbation set: only the explainer varies."""
    t0 = time.perf_counter()
    clean = pipeline.explain(sub, cfg, explainer_name, X)
    repeats = [pipeline.explain(sub, cfg, explainer_name, X)
               for _ in range(n_repeat)]
    B = robustness_B(clean, repeats)
    exact = [bool(np.array_equal(clean, r)) for r in repeats]
    return {"explainer": explainer_name, "n_alerts": int(len(X)),
            "n_repeat": n_repeat,
            "bitwise_identical_repeats": int(sum(exact)),
            "deterministic": all(exact),
            "B_self": summarise("B", B),
            "seconds": round(time.perf_counter() - t0, 1)}


def build(corpus: str, cfg: ExperimentConfig):
    """(substrate, provenance) for the requested corpus."""
    if corpus == "bodmas":
        sub = pipeline.build_substrate(load_bodmas(), cfg, "histgb", cfg.seed,
                                       N_ALERTS)
        return sub, {"corpus": "BODMAS"}
    from txai_exp.ember.substrate import build_ember_substrate
    return build_ember_substrate(cfg, "histgb", cfg.seed, N_ALERTS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="ember",
                        choices=["ember", "bodmas"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = ExperimentConfig()
    sub, provenance = build(args.corpus, cfg)
    X = sub.X_alerts[:N_ALERTS]

    out = {"environment": environment(), "detector": "histgb", "seed": cfg.seed,
           "corpus": args.corpus, "provenance": provenance,
           "purpose": ("Eq. (6) is stated for views deterministic conditional "
                       "on x. This measures whether that holds, per explainer."),
           "by_explainer": {}, "lime_sample_sweep": {}}

    for name in ("treeshap", "constant", "random", "lime"):
        out["by_explainer"][name] = self_stability(sub, cfg, name, X)
        logger.info("%s: B_self mean=%.4f deterministic=%s", name,
                    out["by_explainer"][name]["B_self"]["B_mean"],
                    out["by_explainer"][name]["deterministic"])

    for n_samples in LIME_SAMPLE_GRID:
        swept = replace(cfg, lime_samples=n_samples)
        row = self_stability(sub, swept, "lime", X, n_repeat=3)
        out["lime_sample_sweep"][str(n_samples)] = row
        logger.info("lime %d samples: B_self mean=%.4f (%.0fs)", n_samples,
                    row["B_self"]["B_mean"], row["seconds"])

    save(f"E4b_self_stability_{args.corpus}", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
