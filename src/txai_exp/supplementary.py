"""The four measurements the review asked for, over an already-built substrate.

Each answers one objection, and each is deliberately shaped so that it can come
out against the manuscript:

* `sensitivity_sweep` recomputes the headline admissibility region under three
  disclosure costs and three faithfulness scores -- nine instantiations of
  Definition 3, of which the manuscript reported one. If the constant control
  covers more threshold space than TreeSHAP in all nine, the finding is about
  the definition. If it covers more in one, the finding was about the witness.
* `actionability_block` adds the fourth admissibility condition, so the same
  region is reported over {F, B, D} and over {F, B, D, A}. The gap between the
  two is exactly what the missing condition was worth.
* `calibration_block` sweeps the sufficiency threshold, converting "no map is
  admissible anywhere" into the largest sufficiency each map can promise.
* `controls_block` runs the positive controls.

Everything here takes a `Substrate` and the attributions already computed for
it, so nothing re-fits a detector or re-explains an alert: the expensive work is
done once by the caller and shared.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from . import controls as _controls
from .actionability import ACTION_PLAYBOOK, actionability_A
from .alt_metrics import (disclosure_count, disclosure_mass,
                          faithfulness_flip, faithfulness_relative)
from .calibration_sweep import sweep_theta
from .config import ROLE_ORDER, ExperimentConfig
from .explainers import build_explainer
from .metrics import disclosure_D, faithfulness_F, is_admissible, sufficiency_Phi

__all__ = ["admissible_region", "sensitivity_sweep", "actionability_block",
           "calibration_block", "controls_block"]

logger = logging.getLogger(__name__)

#: The manuscript's headline reads a map as covering a threshold-role
#: combination when it is admissible for more than half the alerts there.
REGION_RATE = 0.5


def _aligned_n(phi: np.ndarray, B: Optional[np.ndarray] = None) -> int:
    """The alert prefix on which F, B and D are jointly defined, per explainer.

    Faithfulness and disclosure are computed on `n_explain` alerts; robustness
    on the first `n_robust` of them, which is smaller. Definition 3 conjoins the
    three, so the region is only defined on the intersection -- the first
    `n_robust` alerts, which is the same prefix E7 uses. Reading the length of
    the attributions alone silently paired a 1000-long F with a 500-long B.

    The prefix is per explainer and *not* shared. LIME runs on a reduced grid
    that its cost forces, so a single minimum over all explainers would drag
    TreeSHAP and both controls down to LIME's alert count -- reporting the
    headline sweep on a fifth of the alerts it was run on, under a label that
    says otherwise. The two arms the verdict compares (constant and TreeSHAP)
    are both on the full grid and so still share a prefix; the returned
    `n_alerts_by_explainer` records the rest.
    """
    n = len(phi)
    if B is not None:
        n = min(n, len(np.asarray(B)))
    return n


def admissible_region(F: np.ndarray, B: np.ndarray,
                      D_by_role: Mapping[str, np.ndarray],
                      cfg: ExperimentConfig,
                      A: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Fraction of the threshold-role grid the map covers, and the mean rate.

    "Covers" means admissible for more than half the alerts, which is the
    convention the manuscript's 28.1% figure uses. The mean rate is reported
    beside it because the fraction is a thresholded summary and can move while
    the underlying rates barely do.

    When `A` is supplied the grid gains no dimension: tau_a is held at the
    configured default rather than swept, so the four-condition region stays
    comparable to the three-condition one cell for cell.
    """
    n_covered = 0
    total = 0
    rate_sum = 0.0
    for role in ROLE_ORDER:
        D = D_by_role[role]
        for tau_f in cfg.threshold_grid:
            for tau_b in cfg.threshold_grid:
                for tau_d in cfg.threshold_grid:
                    ok = is_admissible(F, B, D, tau_f, tau_b, tau_d,
                                       A=A, tau_a=cfg.tau_a if A is not None
                                       else None)
                    rate = float(ok.mean())
                    rate_sum += rate
                    n_covered += int(rate > REGION_RATE)
                    total += 1
    return {"grid_points": total,
            "covered": n_covered,
            "covered_fraction": n_covered / total,
            "mean_rate": rate_sum / total}


def _f_variants(sub, cfg: ExperimentConfig, X: np.ndarray, phi: np.ndarray,
                random_phis: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    """The three faithfulness scores on one explainer's attributions."""
    return {
        "deletion_auc": faithfulness_F(sub.score_fn, X, phi, sub.reference,
                                       steps=cfg.deletion_steps),
        "random_relative": faithfulness_relative(sub.score_fn, X, phi,
                                                 sub.reference, random_phis,
                                                 steps=cfg.deletion_steps),
        "flip_point": faithfulness_flip(sub.score_fn, X, phi, sub.reference),
    }


def _d_variants(phi: np.ndarray, cfg: ExperimentConfig
                ) -> Dict[str, Dict[str, np.ndarray]]:
    """The three disclosure costs, each as a role -> per-alert array map."""
    return {
        "weighted_cover": {r: disclosure_D(phi, r, cfg.top_k)
                           for r in ROLE_ORDER},
        "unweighted_count": {r: disclosure_count(phi, r, cfg.top_k)
                             for r in ROLE_ORDER},
        "mass_weighted": {r: disclosure_mass(phi, r, cfg.top_k)
                          for r in ROLE_ORDER},
    }


def sensitivity_sweep(sub, cfg: ExperimentConfig,
                      attributions: Mapping[str, np.ndarray],
                      B_by_explainer: Mapping[str, np.ndarray],
                      n_random_draws: int = 3) -> dict:
    """The headline region under 3 x 3 instantiations of D and F.

    Args:
        sub: the substrate the attributions were computed on.
        cfg: the frozen run configuration.
        attributions: explainer name -> attributions, all on the same alerts.
        B_by_explainer: explainer name -> robustness per alert, held fixed
            across the sweep because the sweep is about D and F.
        n_random_draws: random control draws averaged into the relative-F floor.

    Returns:
        A dict with the per-cell regions and, for every (D, F) pair, whether the
        constant control still covers more threshold space than TreeSHAP.

    Raises:
        KeyError: when an explainer has attributions but no robustness array,
            which would otherwise be filled with a default and silently make
            the comparison meaningless.
    """
    t0 = time.perf_counter()
    n_by_explainer = {name: _aligned_n(phi, B_by_explainer.get(name))
                      for name, phi in attributions.items()}
    n_max = max(n_by_explainer.values())

    # The floor for the relative F. Drawn from the same control the manuscript
    # already uses, at several seeds so the floor is not one unlucky ordering.
    # Drawn once at the longest prefix and sliced per explainer, so two
    # explainers measured on the same alerts face the same floor.
    random_phis_full = []
    for j in range(n_random_draws):
        # Built directly rather than through `pipeline.explain`, whose cache
        # would hand back the block's own random instance and advance its
        # stream -- which would change the E4 robustness numbers as a side
        # effect of measuring faithfulness.
        expl = build_explainer("random", n_features=sub.X_alerts.shape[1],
                               seed=sub.seed + 1000 + j)
        random_phis_full.append(expl.explain(sub.X_alerts[:n_max]))

    cells = []
    per_explainer: Dict[str, dict] = {}
    for name, phi_full in attributions.items():
        if name not in B_by_explainer:
            raise KeyError(f"no robustness array for {name!r}; the sweep holds "
                           "B fixed and cannot invent it")
        n = n_by_explainer[name]
        X = sub.X_alerts[:n]
        random_phis = [r[:n] for r in random_phis_full]
        phi = phi_full[:n]
        B = np.asarray(B_by_explainer[name])[:n]
        Fs = _f_variants(sub, cfg, X, phi, random_phis)
        Ds = _d_variants(phi, cfg)
        per_explainer[name] = {
            "n_alerts": int(n),
            "F_means": {k: float(v.mean()) for k, v in Fs.items()},
            "D_means": {dk: {r: float(v.mean()) for r, v in dv.items()}
                        for dk, dv in Ds.items()},
        }
        for f_name, F in Fs.items():
            for d_name, D_by_role in Ds.items():
                region = admissible_region(F, B, D_by_role, cfg)
                cells.append({"explainer": name, "F": f_name, "D": d_name,
                              "n_alerts": int(n), **region})

    verdict = []
    for f_name in ("deletion_auc", "random_relative", "flip_point"):
        for d_name in ("weighted_cover", "unweighted_count", "mass_weighted"):
            def _cov(expl: str) -> Optional[float]:
                for c in cells:
                    if (c["explainer"] == expl and c["F"] == f_name
                            and c["D"] == d_name):
                        return c["covered_fraction"]
                return None

            const, tree = _cov("constant"), _cov("treeshap")
            if const is None or tree is None:
                continue
            verdict.append({"F": f_name, "D": d_name,
                            "constant_covered_fraction": const,
                            "treeshap_covered_fraction": tree,
                            "constant_wider": bool(const > tree),
                            "margin": const - tree})

    # The verdict subtracts one covered fraction from another, which is only a
    # comparison if both were measured on the same alerts. Both arms run the
    # full grid, so this holds; it is asserted rather than assumed because a
    # future reduced grid for the constant control would otherwise produce a
    # margin that looks like a finding and is an artefact of sample size.
    if n_by_explainer.get("constant") != n_by_explainer.get("treeshap"):
        raise ValueError(
            "constant and TreeSHAP were measured on different alert counts "
            f"({n_by_explainer.get('constant')} vs "
            f"{n_by_explainer.get('treeshap')}); the covered-fraction margin "
            "would not be a comparison")

    survives = all(v["constant_wider"] for v in verdict) if verdict else None
    logger.info("sensitivity sweep: constant wider in %d of %d instantiations",
                sum(v["constant_wider"] for v in verdict), len(verdict))

    return {
        "n_alerts_by_explainer": {k: int(v) for k, v in n_by_explainer.items()},
        "n_random_draws": int(n_random_draws),
        "F_variants": ["deletion_auc", "random_relative", "flip_point"],
        "D_variants": ["weighted_cover", "unweighted_count", "mass_weighted"],
        "per_explainer": per_explainer,
        "cells": cells,
        "constant_vs_treeshap": verdict,
        "finding_survives_all_instantiations": survives,
        "seconds": round(time.perf_counter() - t0, 1),
    }


def actionability_block(sub, cfg: ExperimentConfig,
                        attributions: Mapping[str, np.ndarray],
                        B_by_explainer: Mapping[str, np.ndarray],
                        perturbed: Optional[Mapping[str, Sequence[np.ndarray]]]
                        = None) -> dict:
    """A_Gamma per explainer, and the region with and without it.

    The comparison is the point: the manuscript's finding was measured over
    three of Definition 3's four conditions, and the omitted one is the only one
    with a reason to reject a map that is identical on every alert.
    """
    t0 = time.perf_counter()
    n_by_explainer = {name: _aligned_n(phi, B_by_explainer.get(name))
                      for name, phi in attributions.items()}

    rows = {}
    for name, phi_full in attributions.items():
        n = n_by_explainer[name]
        X = sub.X_alerts[:n]
        phi = phi_full[:n]
        B = np.asarray(B_by_explainer[name])[:n]
        parts = actionability_A(
            X, phi, sub.background, top_k=cfg.top_k,
            phi_perturbed=([p[:n] for p in perturbed[name]]
                           if perturbed and name in perturbed else None))
        F = faithfulness_F(sub.score_fn, X, phi, sub.reference,
                           steps=cfg.deletion_steps)
        D_by_role = {r: disclosure_D(phi, r, cfg.top_k) for r in ROLE_ORDER}

        without = admissible_region(F, B, D_by_role, cfg)
        with_A = admissible_region(F, B, D_by_role, cfg, A=parts["A"])
        # A_Gamma is a proxy, so a single tau_a would be a tuned number. Swept
        # instead, on the same grid the other three conditions use: the useful
        # statement is the range of tau_a over which the conditions separate the
        # maps, not the region at one setting.
        by_tau_a = {}
        for tau_a in cfg.threshold_grid:
            region = admissible_region(F, B, D_by_role,
                                       replace(cfg, tau_a=float(tau_a)),
                                       A=parts["A"])
            by_tau_a[f"{tau_a:g}"] = region["covered_fraction"]
        groups, counts = np.unique(parts["modal_group"], return_counts=True)
        rows[name] = {
            "n_alerts": int(n),
            "A_mean": float(parts["A"].mean()),
            "A_median": float(np.median(parts["A"])),
            "decisiveness_mean": float(parts["decisiveness"].mean()),
            "grounding_mean": float(parts["grounding"].mean()),
            "stability_mean": float(parts["stability"].mean()),
            "stability_measured": bool(perturbed and name in perturbed),
            "distinct_modal_groups": int(len(groups)),
            "modal_group_counts": {str(g): int(c)
                                   for g, c in zip(groups, counts)},
            "region_without_A": without,
            "region_with_A": with_A,
            "region_with_A_by_tau_a": by_tau_a,
            "region_lost_to_A": (without["covered_fraction"]
                                 - with_A["covered_fraction"]),
        }

    return {
        "tau_a": cfg.tau_a,
        "n_alerts_by_explainer": {k: int(v) for k, v in n_by_explainer.items()},
        "status": ("A_Gamma is instantiated from an illustrative playbook; it "
                   "is a proxy, not a validated operational metric, and is "
                   "reported separately from the three measured conditions"),
        "playbook": {g: {"action": a, "precondition": p}
                     for g, (a, p) in ACTION_PLAYBOOK.items()},
        "per_explainer": rows,
        "seconds": round(time.perf_counter() - t0, 1),
    }


def calibration_block(sub, cfg: ExperimentConfig,
                      attributions: Mapping[str, np.ndarray]) -> dict:
    """Equation (4) relaxed to Phi >= theta, per explainer and release width."""
    t0 = time.perf_counter()
    # Both F and Phi are per-alert, so each explainer is swept on its own alerts
    # rather than on the shortest explainer's. theta* is a per-map promise and
    # is never differenced across maps, so the counts need not share a prefix --
    # but each row carries the count it was computed on.
    n_by_explainer = {name: _aligned_n(phi) for name, phi in attributions.items()}

    rows = []
    for name, phi_full in attributions.items():
        n = n_by_explainer[name]
        X = sub.X_alerts[:n]
        phi = phi_full[:n]
        F = faithfulness_F(sub.score_fn, X, phi, sub.reference,
                           steps=cfg.deletion_steps)
        for k in cfg.top_k_grid:
            Phi = sufficiency_Phi(sub.score_fn, X, phi, sub.reference, top_k=k)
            rows.append({"explainer": name, "top_k": int(k), "n_alerts": int(n),
                         **sweep_theta(F, Phi, cfg.tau_f)})

    frontiers = {r["explainer"]: {} for r in rows}
    for r in rows:
        frontiers[r["explainer"]][int(r["top_k"])] = r["frontier_theta"]

    return {
        "tau_f": cfg.tau_f,
        "n_alerts_by_explainer": {k: int(v) for k, v in n_by_explainer.items()},
        "reading": ("theta* is the largest sufficiency the map can promise at "
                    "this width; theta* = 1 recovers the strict condition, and "
                    "None means no alert cleared tau_f so Equation (4) is "
                    "vacuous rather than satisfied"),
        "frontier_by_explainer_and_top_k": frontiers,
        "sweeps": rows,
        "seconds": round(time.perf_counter() - t0, 1),
    }


def controls_block(sub, cfg: ExperimentConfig, phi: np.ndarray,
                   bridge: Optional[Mapping[str, float]] = None) -> dict:
    """Positive controls, run on TreeSHAP's attributions."""
    t0 = time.perf_counter()
    n = len(phi)
    out = _controls.run_all_controls(
        phi, top_k_grid=cfg.top_k_grid, bridge=bridge, top_k=cfg.top_k,
        score_fn=sub.score_fn, X=sub.X_alerts[:n], reference=sub.reference)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out
