"""Relaxing Equation (4) from Phi = 1 to Phi >= theta, and finding where it bites.

Under the strict reading of the criterion calibration condition -- ``F >= tau_f
implies Phi(z, x) = 1`` -- E3 found violations for every map at every release
width tried, TreeSHAP included. Section V says a nonzero count voids every
admissibility verdict issued at that threshold. Taken together, the contract
certifies nothing on its own instantiation. That is either a finding or a
mis-specification, and the manuscript cannot tell which while only one value of
the sufficiency threshold has been tried.

This module measures the whole family. Two quantities:

* **the violation count as a function of theta** -- how far the sufficiency
  requirement has to be relaxed before the acceptance set is nonempty;
* **the calibration frontier** ``theta*`` -- the exact largest threshold at
  which the map is calibrated, which is just the smallest Phi among the alerts
  the faithfulness condition admits. No search needed, and no grid resolution to
  argue about.

The frontier is the honest summary: it converts "the contract rejects
everything" into a number that says *how much* sufficiency each map can actually
promise. A map whose frontier is 0 promises nothing, and that distinction --
between a map that fails a strict test narrowly and one that fails it
completely -- is invisible in a table of violation counts at Phi = 1.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

__all__ = ["calibration_frontier", "coverage_frontier", "sweep_theta",
           "sweep_theta_tau_f"]

logger = logging.getLogger(__name__)

#: Sufficiency thresholds swept by default. Includes 1.0 so the strict result
#: the manuscript already reports appears in the same table as its relaxations.
DEFAULT_THETA_GRID = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0)

#: Coverage levels for the quantile frontier. 1.0 is the worst-case frontier,
#: included so the two definitions appear side by side.
DEFAULT_COVERAGE = (0.90, 0.95, 0.99, 1.0)


def calibration_frontier(F: np.ndarray, Phi: np.ndarray, tau_f: float
                         ) -> Optional[float]:
    """The largest theta for which ``F >= tau_f => Phi >= theta`` holds.

    Args:
        F: faithfulness per alert.
        Phi: sufficiency rate per alert, in [0, 1].
        tau_f: the faithfulness threshold under test.

    Returns:
        ``min{Phi(z) : F(z) >= tau_f}``, or None when no alert clears tau_f --
        in which case the implication is vacuously true at every theta and
        reporting a frontier of 1.0 would claim a guarantee that rests on an
        empty antecedent.
    """
    selected = F >= tau_f
    if not selected.any():
        logger.info("calibration frontier undefined: no alert reaches "
                    "tau_f = %.3f, so Equation (4) is vacuous", tau_f)
        return None
    return float(Phi[selected].min())


def coverage_frontier(F: np.ndarray, Phi: np.ndarray, tau_f: float,
                      coverage: float) -> Optional[float]:
    """The largest theta that holds on at least `coverage` of the faithful alerts.

    The worst-case frontier is a minimum, so one alert with Phi = 0 sends it to
    zero however well the map does elsewhere -- which is what the first EMBER
    run showed for every map at every width. A deployment does not read a
    contract that way: it wants the sufficiency it can promise on all but a
    stated fraction of releases, with the remainder escalated rather than
    certified.

    This is the ``1 - coverage`` quantile of Phi over the alerts the
    faithfulness condition admits. At ``coverage = 1.0`` it is exactly
    `calibration_frontier`.

    Returns:
        The threshold, or None when the antecedent is empty.
    """
    selected = F >= tau_f
    if not selected.any():
        return None
    return float(np.quantile(Phi[selected], 1.0 - coverage,
                             method="lower"))


def sweep_theta(F: np.ndarray, Phi: np.ndarray, tau_f: float,
                theta_grid: Sequence[float] = DEFAULT_THETA_GRID
                ) -> Dict[str, object]:
    """Violations of the relaxed criterion condition, over a grid of theta.

    Returns a dict carrying, for each theta, the violation count and rate, plus
    the antecedent size and the frontier. The antecedent size is reported
    because a violation count of zero means two very different things depending
    on whether the faithfulness condition admitted 200 alerts or none.
    """
    selected = F >= tau_f
    n_selected = int(selected.sum())
    rows: List[Mapping[str, float]] = []
    for theta in theta_grid:
        violations = int((selected & (Phi < theta)).sum())
        rows.append({
            "theta": float(theta),
            "violations": violations,
            "violation_rate": (violations / n_selected) if n_selected else 0.0,
        })
    return {
        "tau_f": float(tau_f),
        "n_alerts": int(len(F)),
        "n_faithful": n_selected,
        "frontier_theta": calibration_frontier(F, Phi, tau_f),
        "frontier_by_coverage": {
            str(c): coverage_frontier(F, Phi, tau_f, c)
            for c in DEFAULT_COVERAGE},
        "phi_mean_over_faithful": (float(Phi[selected].mean())
                                   if n_selected else None),
        "by_theta": rows,
    }


def sweep_theta_tau_f(F: np.ndarray, Phi: np.ndarray,
                      tau_f_grid: Sequence[float],
                      theta_grid: Sequence[float] = DEFAULT_THETA_GRID
                      ) -> List[Dict[str, object]]:
    """`sweep_theta` over a grid of faithfulness thresholds as well.

    Equation (4) couples the two thresholds, so a violation count reported at
    one tau_f says nothing about the contract at another. Raising tau_f shrinks
    the antecedent and can only lower the violation count -- eventually to zero
    for the uninteresting reason that nothing qualifies. Pairing every row with
    `n_faithful` keeps that failure mode visible instead of letting it read as
    a calibrated contract.
    """
    return [sweep_theta(F, Phi, tau_f, theta_grid) for tau_f in tau_f_grid]
