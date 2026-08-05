"""Alternative instantiations of D and F, for the sensitivity sweep.

The manuscript's headline -- that the numeric admissibility conditions do not
exclude a constant explanation map -- is measured under one disclosure cost
(Eq. 10, a weighted count of the component groups a release reveals) and one
faithfulness score (normalised area under a deletion curve). The manuscript
itself calls the former "an existence witness, not a recommendation". A finding
read off a witness generalises only as far as the witness does.

So this module supplies two further D and two further F, each differing from the
original in a way that could plausibly overturn the result:

* ``disclosure_count`` drops the role weights entirely. If the constant map wins
  only because the weight table happens to price the groups it lands in
  cheaply, this removes the advantage.
* ``disclosure_mass`` charges for *how much* is revealed rather than for which
  groups are touched at all. Eq. (10) is a set cover: revealing one feature of a
  group costs the same as revealing all of it. A map whose mass concentrates in
  one cheap group is charged differently here.
* ``faithfulness_relative`` scores an explanation against a random control on
  the *same alert*, which is what makes it alert-specific. Deleting features in
  any fixed order degrades a confident detector, so the original F assigns a
  constant map credit for degradation that has nothing to do with the alert.
  This subtracts that floor.
* ``faithfulness_flip`` scores by the smallest number of features whose deletion
  flips the decision. It is the quantity `metrics.deletion_curve` already calls
  "the reportable quantity ... in units an analyst understands", measured per
  alert rather than in aggregate.

None of the four is offered as better than the original. The point of the sweep
is that a finding surviving all of them is a property of Definition 3, and a
finding surviving only the original is a property of the witness.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Sequence

import numpy as np

from .config import GROUP_NAMES, ROLE_SENSITIVITY
from .metrics import (_as_reference_bank, _decision_confidence,
                      faithfulness_F, feature_group_index,
                      normalise_attribution)

__all__ = [
    "DISCLOSURE_VARIANTS",
    "FAITHFULNESS_VARIANTS",
    "disclosure_count",
    "disclosure_mass",
    "faithfulness_relative",
    "faithfulness_flip",
    "flip_k_per_alert",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Disclosure variants
# ---------------------------------------------------------------------------

def disclosure_count(phi: np.ndarray, role: str, top_k: int = 20,
                     group_index: np.ndarray | None = None) -> np.ndarray:
    """Unweighted: the fraction of component groups the top-k reveals.

    `role` is accepted and ignored, so this can be swapped for `disclosure_D`
    without the caller special-casing it. Ignoring the role is the whole
    experiment: it asks whether the result depends on the sensitivity table.
    """
    del role
    gidx = feature_group_index() if group_index is None else group_index
    order = np.argsort(-np.abs(phi), axis=1)[:, :top_k]
    revealed = gidx[order]
    counts = np.array([len(np.unique(row)) for row in revealed],
                      dtype=np.float64)
    return counts / len(GROUP_NAMES)


def disclosure_mass(phi: np.ndarray, role: str, top_k: int = 20,
                    group_index: np.ndarray | None = None) -> np.ndarray:
    """Sensitivity-weighted average over where the released mass actually sits.

    Eq. (10) charges w_R(c) as soon as a group is touched at all. Here a group
    is charged in proportion to the share of the released attribution mass it
    carries, so an explanation that merely grazes a sensitive group pays less
    than one that rests on it. Normalised by the largest weight, so the result
    is in [0, 1] and comparable to the original across roles.
    """
    gidx = feature_group_index() if group_index is None else group_index
    weights = np.array([ROLE_SENSITIVITY[g][role] for g in GROUP_NAMES],
                       dtype=np.float64)

    order = np.argsort(-np.abs(phi), axis=1)[:, :top_k]
    n = phi.shape[0]
    rows = np.repeat(np.arange(n), order.shape[1])
    kept = np.zeros_like(phi)
    kept[rows, order.ravel()] = phi[rows, order.ravel()]
    p = normalise_attribution(kept)                       # (n, d), sums to 1

    group_mass = np.zeros((n, len(GROUP_NAMES)), dtype=np.float64)
    np.add.at(group_mass.T, gidx, p.T)
    return (group_mass @ weights) / weights.max()


# ---------------------------------------------------------------------------
# Faithfulness variants
# ---------------------------------------------------------------------------

def faithfulness_relative(score_fn, X: np.ndarray, phi: np.ndarray,
                          reference: np.ndarray,
                          random_phis: Sequence[np.ndarray],
                          steps: int = 20) -> np.ndarray:
    """F measured against a per-alert random floor, in [0, 1].

    ``F_rel = (F(phi) - F_rand) / (1 - F_rand)``, clipped to [0, 1], where
    ``F_rand`` is the mean of the original F over several random attribution
    draws on the same alerts.

    This is the variant the review predicted would matter. On a confident
    detector, deleting features in *any* fixed order lowers confidence, so the
    original F credits an explanation for degradation that carries no
    alert-specific information. Subtracting the random floor removes that credit
    and leaves only what the explanation adds over noise.

    Args:
        score_fn: maps a feature matrix to P(malware).
        X: alerts, shape (n, d).
        phi: attributions under test, shape (n, d).
        reference: benign donor bank.
        random_phis: attribution draws from the random control on the same
            alerts. At least one; more reduces the floor's variance.
        steps: deletion points, passed through to `faithfulness_F`.

    Raises:
        ValueError: when no random draw is supplied, which would make the floor
            undefined and silently reduce this to the original F.
    """
    if not random_phis:
        raise ValueError("faithfulness_relative needs at least one random draw; "
                         "without a floor this is just faithfulness_F")
    F = faithfulness_F(score_fn, X, phi, reference, steps=steps)
    floors = np.stack([faithfulness_F(score_fn, X, r, reference, steps=steps)
                       for r in random_phis])
    floor = floors.mean(axis=0)
    head_room = np.maximum(1.0 - floor, 1e-9)
    return np.clip((F - floor) / head_room, 0.0, 1.0)


def flip_k_per_alert(score_fn, X: np.ndarray, phi: np.ndarray,
                     reference: np.ndarray,
                     k_values: Sequence[int] | None = None) -> np.ndarray:
    """Smallest k whose deletion flips the decision, per alert; inf if never.

    `metrics.deletion_curve` computes this but returns only aggregates. The
    admissibility surface needs it per alert.
    """
    n, d = X.shape
    ks = list(k_values) if k_values is not None else [
        k for k in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000) if k <= d]
    bank = _as_reference_bank(reference, d)
    order = np.argsort(-np.abs(phi), axis=1)
    base = score_fn(X)
    predicted_malware = base >= 0.5

    conf = np.zeros((len(bank), len(ks), n), dtype=np.float64)
    for r, ref in enumerate(bank):
        for j, k in enumerate(ks):
            X_work = X.astype(np.float64, copy=True)
            cols = order[:, :k]
            rows = np.repeat(np.arange(n), k)
            X_work[rows, cols.ravel()] = ref[cols.ravel()]
            conf[r, j] = _decision_confidence(score_fn(X_work),
                                              predicted_malware)

    flipped = conf.mean(axis=0) < 0.5                      # (len(ks), n)
    flip_k = np.full(n, np.inf)
    for j in range(len(ks) - 1, -1, -1):
        flip_k[flipped[j]] = ks[j]
    return flip_k


def faithfulness_flip(score_fn, X: np.ndarray, phi: np.ndarray,
                      reference: np.ndarray,
                      k_values: Sequence[int] | None = None) -> np.ndarray:
    """F from the flip point: 1 - log(1 + k*) / log(1 + d), in [0, 1].

    An explanation that identifies a handful of decisive features scores near 1;
    one whose ordering never flips the decision scores 0. The log scale is
    deliberate -- the difference between flipping at 2 features and at 20 is the
    interesting one, and a linear scale would hide it next to d = 2,381.
    """
    d = X.shape[1]
    flip_k = flip_k_per_alert(score_fn, X, phi, reference, k_values)
    out = 1.0 - np.log1p(flip_k) / np.log1p(d)
    return np.clip(np.where(np.isfinite(flip_k), out, 0.0), 0.0, 1.0)


#: Disclosure instantiations, keyed by the name used in the results file. The
#: original is included so the sweep reports it on the same footing.
DISCLOSURE_VARIANTS: Dict[str, Callable] = {
    "weighted_cover": None,      # metrics.disclosure_D; filled by the caller
    "unweighted_count": disclosure_count,
    "mass_weighted": disclosure_mass,
}

#: Faithfulness instantiations. `relative` and `flip` need extra arguments, so
#: the sweep driver calls them directly rather than through a uniform signature.
FAITHFULNESS_VARIANTS = ("deletion_auc", "random_relative", "flip_point")
