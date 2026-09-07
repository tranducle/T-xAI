"""Positive controls: inject a known defect and show each check catches it.

Three checks in Section VII report zero violations, and the review's objection
is that none of them could have reported anything else. Two of the three are
provable from the definitions -- widening a view adds nonnegative terms, and the
role ordering holds because the weight table was hand-written to be pointwise
decreasing -- so measuring them re-derives an identity. The third, the Theorem 1
bridge, holds on 138 configurations with a median tightness ratio of 0.027,
which is a 37x headroom; agreement at that slack carries little information.

A check that has never failed has not been tested. Each function here breaks one
assumption in a known way and asserts that the check notices. What that buys:

* the bridge diagnostic gets a *falsification margin* -- the factor by which a
  robustness estimate must be overstated before the bound breaks. This turns
  "the bound is loose" from a complaint into a measured quantity;
* the view-monotonicity check gets a weight table with one negative entry, which
  is the exact hypothesis Proposition `lp` needs, and the check must fire;
* the role-ordering check gets a table with two roles transposed, and must fire.
  A second copy of that control walks the *unmodified* table in the transposed
  order public <= auditor <= analyst <= designer. That ordering is what an
  earlier draft of the manuscript asserted while this code used the other one.
  The manuscript has since been corrected to match the code, so this control is
  now a regression guard rather than a diagnostic -- and it is what keeps the
  correction honest: if the check could not tell the two orderings apart, the
  agreement it reports would be a property of the check and not of the table.

The controls are cheap, and none of them changes a production code path: the
weight table is patched inside a context manager and restored on exit, including
on exception.
"""

from __future__ import annotations

import contextlib
import copy
import logging
from typing import Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np

from . import metrics as _metrics
from . import release as _release
from .config import GROUP_NAMES, ROLE_ORDER, ROLE_SENSITIVITY

__all__ = [
    "patched_weights",
    "patched_role_order",
    "reversed_attribution",
    "bridge_falsification_margin",
    "invalid_kernel_control",
    "negative_weight_control",
    "swapped_role_control",
    "paper_lattice_control",
    "anti_faithful_control",
    "run_all_controls",
]

logger = logging.getLogger(__name__)

#: The transposed reading of the lattice: public <= auditor <= analyst <=
#: designer. An earlier draft of the manuscript stated this order while
#: `ROLE_ORDER` in `config` stated the other; the manuscript now matches the
#: code, and this constant is retained as the second transposition control.
#: Do not "fix" it to match `ROLE_ORDER` -- that would make the control vacuous.
TRANSPOSED_ROLE_ORDER = ("public", "auditor", "analyst", "designer")

#: Retained under the old name so a caller pinned to it keeps working.
PAPER_ROLE_ORDER = TRANSPOSED_ROLE_ORDER


# ---------------------------------------------------------------------------
# Patching helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def patched_weights(table: Mapping[str, Mapping[str, float]]) -> Iterator[None]:
    """Temporarily replace the sensitivity table `disclosure_D` reads.

    `metrics` binds ROLE_SENSITIVITY into its own namespace at import, so this
    rebinds that name rather than the one in `config`. Restored in a finally
    block: a control that left a defective table installed would corrupt every
    measurement taken after it in the same process.
    """
    original = _metrics.ROLE_SENSITIVITY
    _metrics.ROLE_SENSITIVITY = table
    try:
        yield
    finally:
        _metrics.ROLE_SENSITIVITY = original


@contextlib.contextmanager
def patched_role_order(order: Sequence[str]) -> Iterator[None]:
    """Temporarily replace the role order `role_disclosure_profile` walks."""
    original = _release.ROLE_ORDER
    _release.ROLE_ORDER = tuple(order)
    try:
        yield
    finally:
        _release.ROLE_ORDER = original


def reversed_attribution(phi: np.ndarray) -> np.ndarray:
    """An attribution whose magnitude ordering is exactly reversed.

    The anti-explanation: it ranks the features the original ranked last. Used
    as a control on the faithfulness measurement -- deleting in this order
    should degrade the decision *less* than deleting in the original order, and
    an F that does not separate the two is not measuring an ordering at all.
    """
    mag = np.abs(phi)
    order = np.argsort(mag, axis=1)                 # ascending
    out = np.zeros_like(mag)
    ranks = np.arange(phi.shape[1], 0, -1, dtype=np.float64)
    for i in range(phi.shape[0]):
        out[i, order[i]] = ranks
    return out


# ---------------------------------------------------------------------------
# Control 1: bridge diagnostic
# ---------------------------------------------------------------------------

def bridge_falsification_margin(bridge: Mapping[str, float],
                                grid: Optional[Sequence[float]] = None
                                ) -> Dict[str, object]:
    """How far a robustness estimate must be overstated before the check fires.

    The bound is ``Adv^TV <= min{1, L_pi E[1 - B]}``. Overstating robustness by
    a factor c means reporting ``E[1 - B] / c`` in place of the measured
    instability, which shrinks the right-hand side. The check fires as soon as
    ``LHS > E[1 - B] / c``, i.e. once ``c > 1 / tightness_ratio``.

    Both the closed form and a grid scan are returned, and they must agree; the
    scan exists because a closed form that is never executed is an assertion,
    and the point of a control is to execute it.

    Args:
        bridge: a `release.bridge_check` result.
        grid: inflation factors to try. Default spans 1x to 1000x.

    Returns:
        dict with the closed-form margin, the smallest grid factor that fires,
        and the per-factor verdicts.
    """
    lhs = float(bridge["lhs_adv_tv"])
    instability = float(bridge["expected_instability"])
    if lhs <= 0.0:
        return {"margin_closed_form": None,
                "reason": "LHS is zero, so no inflation of B can break the "
                          "diagnostic inequality; this configuration has no finite falsification margin",
                "fires_at": None, "scan": []}

    closed_form = instability / lhs
    if grid is not None:
        factors = sorted(float(c) for c in grid)
    else:
        # A coarse grid would report a `fires_at` far above the closed form and
        # make the two look like they disagree, so bracket the prediction.
        factors = sorted({1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0,
                          100.0, 1000.0,
                          round(closed_form * 0.99, 6),
                          round(closed_form * 1.01, 6)})

    scan: List[Dict[str, float]] = []
    fires_at = None
    for c in factors:
        rhs = min(1.0, instability / c)
        holds = lhs <= rhs + 1e-12
        scan.append({"inflation": float(c), "rhs": float(rhs),
                     "holds": bool(holds)})
        if not holds and fires_at is None:
            fires_at = float(c)

    return {
        "lhs_adv_tv": lhs,
        "expected_instability": instability,
        "margin_closed_form": float(closed_form),
        "fires_at": fires_at,
        # Agreement means every factor at or below the prediction held and the
        # first one above it did not -- not merely that something eventually
        # fired, which a coarse grid would satisfy for the wrong reason.
        "scan_agrees_with_closed_form": bool(
            fires_at is not None
            and fires_at >= closed_form
            and all(row["holds"] for row in scan
                    if row["inflation"] <= closed_form)),
        "scan": scan,
    }


def invalid_kernel_control() -> Dict[str, object]:
    """A kernel that is not column-stochastic must be rejected, not measured.

    L_pi = 1 for the bridge calculation because a column-stochastic matrix is an l1
    non-expansion. A matrix whose columns sum to more than 1 amplifies, so the
    calculation with L_pi = 1 is unsound -- and `ResponseKernel` must refuse it
    rather than silently reporting a Lipschitz constant of 1.
    """
    from .config import ACTIONS

    G = np.zeros((len(ACTIONS), len(GROUP_NAMES)), dtype=np.float64)
    G[0, :] = 1.5                                   # column sums 1.5, not 1
    try:
        _release.ResponseKernel(G=G, name="amplifying")
    except ValueError as exc:
        return {"check": "kernel column-stochasticity", "caught": True,
                "error": str(exc)}
    return {"check": "kernel column-stochasticity", "caught": False,
            "error": "an amplifying kernel was accepted; L_pi = 1 is unsound"}


# ---------------------------------------------------------------------------
# Control 2: view monotonicity
# ---------------------------------------------------------------------------

def negative_weight_control(phi: np.ndarray, top_k_grid: Sequence[int],
                            group: str = "strings") -> Dict[str, object]:
    """One negative weight, and the view-monotonicity check must fire.

    Proposition `lp` needs ``w_R >= 0``: widening a release adds groups, and the
    cost rises only because every added term is nonnegative. Make one term
    negative and revealing more can cost less, which is precisely the
    least-privilege failure the check is supposed to detect.
    """
    table = copy.deepcopy({g: dict(v) for g, v in ROLE_SENSITIVITY.items()})
    for role in ROLE_ORDER:
        table[group][role] = -1.0

    with patched_weights(table):
        result = _release.check_view_monotonicity(phi, top_k_grid)

    return {
        "check": "view monotonicity (Prop. lp)",
        "defect": f"w_R({group}) set to -1.0 for every role",
        "fired": not result["monotone"],
        "violations": int(result["total_violations"]),
        "mean_D_by_top_k": result["mean_D_by_top_k"],
    }


# ---------------------------------------------------------------------------
# Control 3: role cost ordering
# ---------------------------------------------------------------------------

def _raw_monotone(phi: np.ndarray, top_k: int) -> Dict[str, object]:
    profile = _release.role_disclosure_profile(phi, top_k)
    return {
        "raw_monotone": bool(profile["raw_monotone"]),
        "raw_violations": int(profile["raw_violations"]),
        "raw_pairs": profile["raw_pairs"],
        "mean_raw_D_by_role": profile["mean_raw_D_by_role"],
    }


def swapped_role_control(phi: np.ndarray, top_k: int = 20) -> Dict[str, object]:
    """Transpose two roles in the weight table; the ordering check must fire.

    The table is hand-written pointwise decreasing in privilege, so the check
    passes by construction. Swapping the analyst and auditor columns is the
    smallest edit that contradicts the lattice while leaving the table otherwise
    intact -- and it is the same defect as the real one, which makes this
    control double as a demonstration that the check was capable of catching it.
    """
    table = {g: dict(v) for g, v in ROLE_SENSITIVITY.items()}
    for g in GROUP_NAMES:
        table[g]["analyst"], table[g]["auditor"] = (
            table[g]["auditor"], table[g]["analyst"])

    with patched_weights(table):
        result = _raw_monotone(phi, top_k)

    return {
        "check": "cross-role cost ordering",
        "defect": "analyst and auditor weight columns transposed",
        "fired": not result["raw_monotone"],
        **result,
    }


def paper_lattice_control(phi: np.ndarray, top_k: int = 20) -> Dict[str, object]:
    """The unmodified table, checked against the transposed lattice ordering.

    Not a synthetic defect: the weight table is left exactly as shipped, and
    only the order the check walks is replaced by the transposed one
    (public <= auditor <= analyst <= designer). This began as a diagnostic --
    an earlier draft of the manuscript stated that order while the code stated
    `ROLE_ORDER`, and this control is how the mismatch was confirmed. The
    manuscript now matches the code, so the control's job has changed: it must
    still fire, because a check that returned "monotone" under both orderings
    would be insensitive to the very ordering it claims to verify, and the
    agreement reported for `ROLE_ORDER` would then be worth nothing.
    """
    with patched_role_order(TRANSPOSED_ROLE_ORDER):
        result = _raw_monotone(phi, top_k)

    return {
        "check": "cross-role cost ordering under the transposed lattice",
        "defect": "none; the weight table is unmodified",
        "role_order_used": list(TRANSPOSED_ROLE_ORDER),
        "role_order_in_code": list(ROLE_ORDER),
        "fired": not result["raw_monotone"],
        **result,
    }


# ---------------------------------------------------------------------------
# Control 4: the faithfulness measurement itself
# ---------------------------------------------------------------------------

def anti_faithful_control(score_fn, X: np.ndarray, phi: np.ndarray,
                          reference: np.ndarray, steps: int = 20
                          ) -> Dict[str, object]:
    """Reverse the ranking; F must drop.

    If F assigns the anti-explanation a score comparable to the explanation, it
    is measuring how much deletion hurts rather than whether the *ordering* is
    right -- which is the review's charge against the constant control's 0.7136.
    """
    F = _metrics.faithfulness_F(score_fn, X, phi, reference, steps=steps)
    F_anti = _metrics.faithfulness_F(score_fn, X, reversed_attribution(phi),
                                     reference, steps=steps)
    return {
        "check": "faithfulness separates an ordering from its reverse",
        "F_mean": float(F.mean()),
        "F_reversed_mean": float(F_anti.mean()),
        "separation": float(F.mean() - F_anti.mean()),
        "fired": bool(F.mean() > F_anti.mean()),
    }


def run_all_controls(phi: np.ndarray, top_k_grid: Sequence[int],
                     bridge: Optional[Mapping[str, float]] = None,
                     top_k: int = 20,
                     score_fn=None, X: Optional[np.ndarray] = None,
                     reference: Optional[np.ndarray] = None
                     ) -> Dict[str, object]:
    """Every control, with a single verdict field for the results table.

    `all_fired` is True when every check caught its injected defect. It excludes
    `paper_lattice`, whose "defect" is a transposed reading of the lattice
    rather than an altered table; that control is reported and read separately,
    because a caller who silently folded it in would be unable to tell a broken
    weight table from a reconciled manuscript.
    """
    out: Dict[str, object] = {
        "kernel_validation": invalid_kernel_control(),
        "view_monotonicity": negative_weight_control(phi, top_k_grid),
        "role_ordering": swapped_role_control(phi, top_k),
        "paper_lattice": paper_lattice_control(phi, top_k),
    }
    if bridge is not None:
        out["bridge_margin"] = bridge_falsification_margin(bridge)
    if score_fn is not None and X is not None and reference is not None:
        out["faithfulness"] = anti_faithful_control(score_fn, X, phi, reference)

    injected = ["kernel_validation", "view_monotonicity", "role_ordering"]
    verdicts = []
    for key in injected:
        entry = out[key]
        verdicts.append(bool(entry.get("fired", entry.get("caught", False))))
    if "faithfulness" in out:
        verdicts.append(bool(out["faithfulness"]["fired"]))
    out["all_fired"] = all(verdicts)
    return out
