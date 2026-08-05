"""E5b -- where does Theorem 1's slack come from, and can the constant be tightened?

Run 2 measures LHS/RHS ratios with a median of 0.029 for TreeSHAP: the bound
holds everywhere and is ~34x loose. Two candidate sources:

  (a) the release kernel genuinely washes out attribution perturbation, or
  (b) the constant $L_\\pi$ is generic rather than specific to the kernel.

The recorded rows separate them. RHS is *identical* across kernel temperatures
0.25 / 1.0 / 4.0 while LHS falls by ~8x across the same range -- so at least
part of the slack is (b): $L_\\pi = 1$ cannot see the temperature, because it is
the l1 non-expansiveness bound that holds for *every* column-stochastic matrix.

The tight constant for a column-stochastic map under TV is its Dobrushin
ergodic coefficient

    delta(W) = max_{j,k} 0.5 * || W[:,j] - W[:,k] ||_1  <= 1,

and the manuscript's kernel is pi(.|z) = G g(z) = (G A) p, with A the
group-aggregation matrix (column-stochastic: each feature lies in exactly one
group). Since the columns of G A are the columns of G repeated, delta(G A) =
delta(G). The Theorem 1 derivation goes through verbatim with delta in place of
L_pi, because

    0.5*||q - q'||_1 <= sum_i P(i) E[ 0.5*||pi - pi'||_1 | i ]
                     <= delta * E_x[ d_Z ] <= delta * E_x[1 - B].

This script (1) computes delta for every kernel the programme used, (2) checks
the contraction empirically rather than trusting the algebra, and (3) re-derives
the bound from the recorded rows with delta substituted, reporting how much of
the measured slack the tighter constant removes and whether any configuration
is violated under it.

No re-explanation, so it costs seconds.

The E5 rows come either from the per-explainer files `run_scoring_layer.py` writes
or, with `--source`, from the single nested file `run_numerical_instantiation.py` writes. The
recomputation is identical; only where the rows are read from differs, so the
EMBER-2018 and BODMAS sharpenings are computed by the same code.

Usage:
    python3 run_kernel_constant.py
    python3 run_kernel_constant.py --source section7_ember --tag ember
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_scoring_layer import RESULTS_DIR, save  # noqa: E402
from txai_exp.config import ACTIONS, GROUP_NAMES  # noqa: E402
from txai_exp.pipeline import KERNEL_TEMPERATURES  # noqa: E402
from txai_exp.release import ResponseKernel, sharp_kernel  # noqa: E402

logger = logging.getLogger("kernel")


def dobrushin(W: np.ndarray) -> float:
    """max pairwise TV between columns; the tight TV->TV Lipschitz constant."""
    n_cols = W.shape[1]
    best = 0.0
    for j in range(n_cols):
        for k in range(j + 1, n_cols):
            best = max(best, 0.5 * float(np.abs(W[:, j] - W[:, k]).sum()))
    return best


def check_contraction_empirically(kernel: ResponseKernel, delta: float,
                                  n_pairs: int = 2000, seed: int = 0) -> dict:
    """Does TV(Gp, Gp') <= delta * TV(p, p') hold on random group laws?

    The algebra says yes; a violation here would mean the constant is wrong and
    every number derived from it is void. Worst observed ratio is reported so a
    near-miss is visible rather than rounded away.
    """
    rng = np.random.default_rng(seed)
    G = kernel.G
    n_groups = G.shape[1]
    p = rng.dirichlet(np.ones(n_groups), size=n_pairs)
    q = rng.dirichlet(np.ones(n_groups), size=n_pairs)
    d_in = 0.5 * np.abs(p - q).sum(axis=1)
    d_out = 0.5 * np.abs((p - q) @ G.T).sum(axis=1)
    ratio = d_out / np.maximum(d_in, 1e-12)
    return {"max_observed_ratio": float(ratio.max()),
            "delta": delta,
            "violations": int((ratio > delta + 1e-9).sum()),
            "n_pairs": n_pairs}


def _e5_rows(source: str | None) -> dict:
    """explainer -> E5 rows, from either result layout.

    Returns only the explainers actually present: an absent explainer is
    skipped, never defaulted, so a missing arm cannot enter the sharpening as
    if it had been measured.
    """
    if source:
        path = RESULTS_DIR / f"{source}.json"
        if not path.exists():
            raise SystemExit(f"missing result file: {path}")
        blocks = json.loads(path.read_text())["blocks"]
        return {name: blk["E5_bridge"]["rows"] for name, blk in blocks.items()}

    out = {}
    for name in ("treeshap", "lime", "random", "constant"):
        path = RESULTS_DIR / f"block_{name}.json"
        if path.exists():
            out[name] = json.loads(path.read_text())["E5_bridge"]["rows"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None,
                        help="a nested result file written by run_numerical_instantiation.py, "
                             "without the .json suffix")
    parser.add_argument("--tag", default=None,
                        help="suffix for the output file, so a second corpus "
                             "cannot overwrite the first")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    kernels = {}
    for temperature in KERNEL_TEMPERATURES:
        kernel = sharp_kernel(seed=0, temperature=temperature)
        delta = dobrushin(kernel.G)
        kernels[str(temperature)] = {
            "kernel": kernel.name,
            "L_pi_manuscript": kernel.lipschitz_constant,
            "dobrushin_delta": delta,
            "empirical_check": check_contraction_empirically(kernel, delta),
        }
        logger.info("T=%.2f  L_pi=1  delta=%.6f", temperature, delta)

    out = {"n_actions": len(ACTIONS), "n_groups": len(GROUP_NAMES),
           "kernels": kernels, "by_explainer": {}}

    for name, rows in _e5_rows(args.source).items():
        recomputed, ratios_old, ratios_new, violations = [], [], [], 0
        for row in rows:
            delta = kernels[str(row["temperature"])]["dobrushin_delta"]
            rhs_new = min(1.0, delta * row["expected_instability"])
            holds = row["lhs_adv_tv"] <= rhs_new + 1e-12
            violations += (not holds)
            recomputed.append({
                "family": row["family"], "strength": row["strength"],
                "temperature": row["temperature"],
                "lhs_adv_tv": row["lhs_adv_tv"],
                "rhs_manuscript": row["rhs_bound"], "rhs_dobrushin": rhs_new,
                "holds_under_dobrushin": holds})
            if row["rhs_bound"] > 0:
                ratios_old.append(row["lhs_adv_tv"] / row["rhs_bound"])
            if rhs_new > 0:
                ratios_new.append(row["lhs_adv_tv"] / rhs_new)

        out["by_explainer"][name] = {
            "n_configurations": len(rows),
            "violations_under_dobrushin": violations,
            "tightness_median_manuscript":
                float(np.median(ratios_old)) if ratios_old else None,
            "tightness_median_dobrushin":
                float(np.median(ratios_new)) if ratios_new else None,
            "slack_factor_removed":
                float(np.median(ratios_new) / np.median(ratios_old))
                if ratios_old and ratios_new else None,
            "rows": recomputed,
        }
        logger.info("%s: tightness %s -> %s (violations %d)", name,
                    out["by_explainer"][name]["tightness_median_manuscript"],
                    out["by_explainer"][name]["tightness_median_dobrushin"],
                    violations)

    out["source"] = args.source or "block_*.json (run_scoring_layer.py)"
    save(f"E5b_kernel_constant{f'_{args.tag}' if args.tag else ''}", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
