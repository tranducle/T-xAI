"""E8 -- compute a strong Stackelberg equilibrium for one finite release game.

Round-2 review (M-8) put the choice plainly: either compute the equilibrium for
one finite instance, or downgrade the release game from a contribution to a
formulation. This script takes the first option.

The instance is built on the same substrate as E1-E7 (BODMAS, histgb, seed 42,
TreeSHAP). The defender chooses how much to release to an untrusted `public`
receiver and which response kernel to act under; the adversary chooses a
capability and a transformation that capability can realise. Every payoff term
that the substrate can evaluate is measured, and the four that cannot -- the
decision-loss matrix and the weights `beta`, `kappa`, `eta`, `rho` -- are
declared in `game.GameInstance` and then swept, because an equilibrium quoted
at one unguided parameter setting answers M-8 no better than existence does.

Two solvers run on the same payoff tables: the DOBSS mixed-integer program via
`scipy.optimize.milp`, and the multiple-LP method as an independent check. They
share no code path. Disagreement beyond 1e-6 aborts the run.

Usage: python3 run_release_game.py
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_scoring_layer import environment, save  # noqa: E402
from txai_exp import pipeline  # noqa: E402
from txai_exp.config import ExperimentConfig  # noqa: E402
from txai_exp.data import load_bodmas  # noqa: E402
from txai_exp.game import (GameInstance, build_payoffs, solve_sse_milp,  # noqa: E402
                           solve_sse_multiple_lp, solve_sse_pure)
from txai_exp.perturbations import apply_perturbation  # noqa: E402
from txai_exp.release import sharp_kernel  # noqa: E402

logger = logging.getLogger("e8")

N_ALERTS = 200
EXPLAINER = "treeshap"
KERNEL_TEMPERATURES = (0.25, 1.0, 4.0)      # the same three E5 measures
CAPABILITY_FAMILY = {"M1_append": "append_bytes",
                     "M1_imports": "add_imports",
                     "M1_generic": "generic"}
RHO_SWEEP = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
ETA_SWEEP = (0.0, 1.0, 4.0)


def build_attacker_moves(sub, cfg: ExperimentConfig
                         ) -> Tuple[List[Tuple[str, str, np.ndarray]],
                                    Dict[str, np.ndarray]]:
    """K = {(m, delta)}: one seeded transformation per capability and budget.

    `Delta_m` in Definition `game` is a whole transformation set; a computable
    instance needs a finite subset of it. The subset chosen here is one draw per
    budget level, seeded from the substrate so the same delta is reproduced on
    every run. This is a restriction of the game, not a claim about it: a wider
    `Delta_m` can only raise the adversary's value.
    """
    X = sub.X_alerts[:N_ALERTS]
    moves: List[Tuple[str, str, np.ndarray]] = []
    X_by_move: Dict[str, np.ndarray] = {}

    # M4: observe the released view, transform nothing.
    phi_clean = pipeline.explain(sub, cfg, EXPLAINER, X)
    moves.append(("M4_observe|id", "M4_observe", phi_clean))
    X_by_move["M4_observe|id"] = X

    for cap, family in CAPABILITY_FAMILY.items():
        for budget in cfg.perturb_budget_grid:
            rng = np.random.default_rng(sub.seed)
            X_p = apply_perturbation(family, X, rng, budget)
            label = f"{cap}|{family}@{budget:g}"
            moves.append((label, cap, pipeline.explain(sub, cfg, EXPLAINER, X_p)))
            X_by_move[label] = X_p
            logger.info("attacker move %s prepared", label)
    return moves, X_by_move


def solve_and_check(tables) -> dict:
    """Both solvers plus the pure-strategy baseline, with agreement enforced."""
    pure = solve_sse_pure(tables)
    mixed = solve_sse_milp(tables)
    check = solve_sse_multiple_lp(tables)
    gap = abs(mixed["defender_utility"] - check["defender_utility"])
    if gap > 1e-6:
        raise SystemExit(
            f"MILP and multiple-LP disagree by {gap:.3e}: "
            f"{mixed['defender_utility']} vs {check['defender_utility']}. "
            "One of the two formulations is wrong; do not report either.")
    if mixed["defender_utility"] < pure["defender_utility"] - 1e-9:
        raise SystemExit(
            "mixed optimum below the pure optimum, which is impossible: "
            "a pure strategy is a feasible mixed strategy.")
    return {"pure": pure, "mixed": mixed,
            "cross_check_multiple_lp": check,
            "milp_vs_lp_gap": gap,
            "randomisation_gain": mixed["defender_utility"]
                                  - pure["defender_utility"]}


def run_e8(sub, cfg: ExperimentConfig) -> dict:
    """The whole of E8 on an already-built substrate.

    Split out from `main` so the EMBER-2018 rerun can drive the same game code
    from a different corpus. Nothing below the substrate is corpus-specific --
    which is the point, since a game solved by a second code path would not be
    comparable to the first.
    """
    t0 = time.perf_counter()
    moves, X_by_move = build_attacker_moves(sub, cfg)
    kernels = [sharp_kernel(seed=cfg.seed, temperature=t)
               for t in KERNEL_TEMPERATURES]
    base = GameInstance()

    tables = build_payoffs(base, kernels, moves, sub.y_alerts[:N_ALERTS],
                           sub.score_fn, X_by_move, sub.reference)
    logger.info("payoff tables: %d defender x %d attacker strategies",
                *tables.U_D.shape)

    equilibrium = solve_and_check(tables)
    logger.info("SSE: defender=%s attacker=%s U_D=%.4f",
                equilibrium["mixed"]["defender_mixed_strategy"],
                equilibrium["mixed"]["attacker_strategy"],
                equilibrium["mixed"]["defender_utility"])

    # Sensitivity: the equilibrium as a function of the two weights the
    # manuscript leaves free. Re-assembling the payoffs per setting is cheap;
    # the explanations, which are the whole cost, are already computed.
    sweep = []
    for rho in RHO_SWEEP:
        for eta in ETA_SWEEP:
            inst = replace(base, rho=rho, eta=eta)
            t = build_payoffs(inst, kernels, moves, sub.y_alerts[:N_ALERTS],
                              sub.score_fn, X_by_move, sub.reference)
            eq = solve_and_check(t)
            sweep.append({
                "rho": rho, "eta": eta,
                "defender_mixed_strategy": eq["mixed"]["defender_mixed_strategy"],
                "attacker_strategy": eq["mixed"]["attacker_strategy"],
                "defender_utility": eq["mixed"]["defender_utility"],
                "pure_defender_strategy": eq["pure"]["defender_strategy"],
                "randomisation_gain": eq["randomisation_gain"],
            })
    logger.info("sensitivity sweep: %d settings solved", len(sweep))

    released = sorted({int(row["pure_defender_strategy"].split(",")[0]
                           .split("=")[1]) for row in sweep})

    out = {
        "environment": environment(),
        "detector": sub.detector_name, "seed": cfg.seed, "explainer": EXPLAINER,
        "n_alerts": N_ALERTS,
        "purpose": ("M-8: compute the equilibrium for one finite instance "
                    "rather than only certifying that one exists."),
        "instance": {
            "loss": {a: list(v) for a, v in base.loss.items()},
            "kappa": dict(base.kappa), "beta": dict(base.beta),
            "eta": base.eta, "rho": base.rho,
            "view_grid": list(base.view_grid),
            "public_ratio": base.public_ratio,
            "public_width": {k: base.public_width(k) for k in base.view_grid},
            "role_defender": base.role_defender,
            "role_attacker": base.role_attacker,
            "kernel_temperatures": list(KERNEL_TEMPERATURES),
        },
        "defender_strategies": list(tables.defender_labels),
        "attacker_strategies": list(tables.attacker_labels),
        "U_D": tables.U_D.tolist(),
        "U_A": tables.U_A.tolist(),
        "measured": tables.measured,
        "equilibrium": equilibrium,
        "sensitivity": sweep,
        "release_levels_chosen_across_sweep": released,
        "world_mass": {
            "benign": float((sub.y_alerts[:N_ALERTS] == 0).mean()),
            "malware": float((sub.y_alerts[:N_ALERTS] == 1).mean()),
        },
        "seconds": round(time.perf_counter() - t0, 1),
    }
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    data = load_bodmas()
    cfg = ExperimentConfig()
    sub = pipeline.build_substrate(data, cfg, "histgb", cfg.seed, N_ALERTS)
    save("E8_release_game", run_e8(sub, cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
