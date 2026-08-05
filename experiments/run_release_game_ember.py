"""Re-run the release game on EMBER-2018, recording the marginal counterfactual.

``run_numerical_instantiation.py --corpus ember`` already solves this game, and this driver
solves the identical instance on the identical substrate: same corpus, same
detector, same seed, same alert population. It exists because the payoff table
gained two fields after that run -- the decision loss the *marginal* formula
would have returned, and its gap from the joint-law value -- and those are the
numbers Proposition `marginal` is about.

The gap is identically zero whenever the latent world is degenerate, which is
what the BODMAS instance was and what the round-3 review objected to. Measuring
it on a two-valued world is the point; measuring it a second time is cheaper
than trusting that it would have been nonzero.

The equilibrium is recomputed as a consistency check, not as a new result: it
must reproduce the one in ``section7_ember.json`` exactly, and a difference
would mean the substrate is not the one the section reports.

Usage:
    python3 run_release_game_ember.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_release_game import run_e8  # noqa: E402
from run_scoring_layer import save  # noqa: E402
from txai_exp.config import ExperimentConfig  # noqa: E402
from txai_exp.ember.substrate import build_ember_substrate  # noqa: E402

logger = logging.getLogger("e8_ember")


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    t0 = time.perf_counter()

    cfg = ExperimentConfig()
    sub, provenance = build_ember_substrate(cfg, "histgb", cfg.seed,
                                            cfg.n_explain)
    logger.info("substrate: AUC=%.6f, %d alerts, %.4f malware",
                sub.auc, len(sub.X_alerts), float((sub.y_alerts == 1).mean()))

    out = run_e8(sub, cfg)
    out["provenance"] = provenance
    out["total_seconds"] = round(time.perf_counter() - t0, 1)

    cells = out["measured"]["cells"]
    gaps = [abs(c["marginal_gap"]) for c in cells]
    logger.info("marginal gap over %d cells: min %.6f median %.6f max %.6f",
                len(gaps), min(gaps), sorted(gaps)[len(gaps) // 2], max(gaps))

    save("E8_release_game_ember", out)
    logger.info("done in %.1f min", out["total_seconds"] / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
