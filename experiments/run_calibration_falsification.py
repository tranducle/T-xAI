"""Re-run E3 alone, recording both readings of the Eq. (7) violation count.

Eq. (7) asks for $\\Phi(z,x)=1$. With a bank of benign donors, $\\Phi$ is the
*fraction* of donors under which the top-$k$ alone reproduces the decision, so
"violation" has two defensible readings:

  strict         F >= tau_f and Phi < 1   -- fails under at least one donor
  total failure  F >= tau_f and Phi == 0  -- fails under every donor

Run 2 recorded only the second. For TreeSHAP at k=50 they differ by 62x (62 vs
1), which is the difference between "the condition is nearly satisfied" and "the
condition still bites". Both belong in the write-up.

Rebuilds the same substrate (seed 42, histgb) and re-explains; ~4 min, dominated
by the detector fit and LIME.

Usage: python3 run_calibration_falsification.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_scoring_layer import environment, save  # noqa: E402
from txai_exp import pipeline  # noqa: E402
from txai_exp.config import ExperimentConfig  # noqa: E402
from txai_exp.data import load_bodmas  # noqa: E402

logger = logging.getLogger("e3")

# The substrate must be built with `n_explain`, not with 200: `build_substrate`
# draws its own random sample of alerts, so asking it for 200 gives a *different*
# 200 than the first 200 of run 2's 1,000-alert draw, and the counts would not be
# comparable to `E3_calibration.json`. Build the same 1,000 and slice.
N_ALERTS = 200          # the cross-explainer subset used by run 2's E2/E3/E6


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    t0 = time.perf_counter()
    data = load_bodmas()
    cfg = ExperimentConfig()
    sub = pipeline.build_substrate(data, cfg, "histgb", cfg.seed, cfg.n_explain)
    X = sub.X_alerts[:N_ALERTS]

    attributions = {name: pipeline.explain(sub, cfg, name, X)
                    for name in ("treeshap", "lime", "random", "constant")}

    out = pipeline.e3_calibration(sub, cfg, attributions)
    out["environment"] = environment()
    out["n_alerts"] = N_ALERTS
    out["note"] = ("Supersedes the E3 block of run 2 for violation counting: "
                   "adds violations_strict (Phi < 1) alongside the "
                   "total-failure count (Phi == 0) that run 2 recorded alone. "
                   "Same substrate, same seed, same alerts.")
    out["seconds"] = round(time.perf_counter() - t0, 1)
    save("E3_calibration_strict", out)

    for name, rows in out["by_explainer_and_top_k"].items():
        for k, r in rows.items():
            logger.info("%-9s k=%-3s strict=%3d total_failure=%3d  "
                        "Phi: =1 %.3f  =0 %.3f  median %.3f", name, k,
                        r["violations_strict"], r["F_passes_but_insufficient"],
                        r["phi_eq_1_rate"], r["phi_eq_0_rate"], r["phi_median"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
