"""The framework's scoring layer: d_Z, B (Eq. 6), F, Phi, D (Eq. 10), admissibility.

Every score is constructed to land in [0, 1] so the thresholds tau_f, tau_a,
tau_b, tau_d of Definition 3 are comparable across alerts. The choices that
carry the argument:

* **d_Z is total variation between normalised attribution masses.** It is a
  genuine metric, it is bounded by 1 without clipping, and -- decisively for
  Theorem 1 -- it is the same divergence the response kernel is measured in, so
  the kernel's Lipschitz constant comes out exact rather than estimated
  (see `release.ResponseKernel`).

* **F and Phi must not be the same test.** Equation (7) asks whether
  F >= tau_f implies Phi = 1. Scoring both with the same procedure would make
  the implication true by construction and the experiment worthless. F is a
  *deletion* curve: remove the most-attributed features and watch confidence
  fall. Phi is an *insertion* test: keep only the top-k and ask whether the
  decision survives. Different direction, different statistic.

* **Confidence is measured on the side the detector actually chose.** Using
  P(malware) directly makes the deletion curve's normaliser near zero for
  benign alerts. Using confidence in the predicted class bounds it below by
  0.5, so the normalisation cannot blow up.
"""

from __future__ import annotations

import logging
from typing import Callable, Mapping, Sequence

import numpy as np

from .config import EMBER_GROUPS, GROUP_NAMES, ROLE_SENSITIVITY

__all__ = [
    "normalise_attribution",
    "attribution_distance",
    "robustness_B",
    "faithfulness_F",
    "sufficiency_Phi",
    "disclosure_D",
    "feature_group_index",
    "is_admissible",
    "deletion_curve",
    "summarise",
]

logger = logging.getLogger(__name__)

ScoreFn = Callable[[np.ndarray], np.ndarray]  # X -> P(malware), shape (n,)

_EPS = 1e-12


def normalise_attribution(phi: np.ndarray) -> np.ndarray:
    """Map attributions to a probability distribution over features.

    Sign is discarded: disclosure and stability are about *which* components an
    explanation leans on, not the direction each pushes. An all-zero
    explanation carries no information about any feature, so it maps to the
    uniform distribution -- the honest representation of "says nothing" -- and
    not to a degenerate point mass.

    Args:
        phi: attributions, shape (..., d).

    Returns:
        Non-negative array summing to 1 along the last axis.
    """
    mass = np.abs(np.asarray(phi, dtype=np.float64))
    total = mass.sum(axis=-1, keepdims=True)
    d = mass.shape[-1]
    return np.where(total > _EPS, mass / np.maximum(total, _EPS), 1.0 / d)


def attribution_distance(phi_a: np.ndarray, phi_b: np.ndarray) -> np.ndarray:
    """d_Z: total variation between two normalised attribution masses, in [0, 1].

    Args:
        phi_a, phi_b: attributions, shape (..., d), broadcastable.

    Returns:
        Distance per leading index, shape (...).
    """
    p = normalise_attribution(phi_a)
    q = normalise_attribution(phi_b)
    return 0.5 * np.abs(p - q).sum(axis=-1)


def robustness_B(phi_clean: np.ndarray,
                 phi_perturbed: Sequence[np.ndarray]) -> np.ndarray:
    """Equation (6): B = 1 - sup over the perturbation set of d_Z.

    The supremum is taken over the sampled set, so this is an *upper* estimate
    of B: a larger perturbation sample can only lower it. Reported as such --
    it is not a certified lower bound on robustness.

    Args:
        phi_clean: attributions on the unperturbed inputs, shape (n, d).
        phi_perturbed: one (n, d) array per perturbation in the sample.

    Returns:
        B per alert, shape (n,), in [0, 1].
    """
    if not phi_perturbed:
        raise ValueError("robustness_B needs at least one perturbation; an "
                         "empty set would report perfect robustness")
    worst = np.zeros(len(phi_clean), dtype=np.float64)
    for phi_d in phi_perturbed:
        worst = np.maximum(worst, attribution_distance(phi_clean, phi_d))
    return 1.0 - worst


def _decision_confidence(scores: np.ndarray, predicted_malware: np.ndarray
                         ) -> np.ndarray:
    """Confidence in the originally predicted class; bounded below by 0.5 at j=0."""
    return np.where(predicted_malware, scores, 1.0 - scores)


def _as_reference_bank(reference: np.ndarray, d: int) -> np.ndarray:
    """Accept a single reference vector or a bank of them, return shape (R, d).

    A bank is the defensible choice and a single vector is kept only for tests.
    See `faithfulness_F` for why the identity of the reference is not a detail.
    """
    ref = np.asarray(reference, dtype=np.float64)
    if ref.ndim == 1:
        ref = ref[None, :]
    if ref.ndim != 2 or ref.shape[1] != d:
        raise ValueError(f"reference must be (d,) or (R, d) with d={d}, "
                         f"got {ref.shape}")
    return ref


def faithfulness_F(score_fn: ScoreFn,
                   X: np.ndarray,
                   phi: np.ndarray,
                   reference: np.ndarray,
                   steps: int = 20) -> np.ndarray:
    """F: normalised area under the deletion curve, in [0, 1], higher = faithful.

    Features are removed in descending |attribution| order by overwriting them
    with a reference value. A faithful explanation collapses the decision
    quickly; an unfaithful one does not. The random-attribution control in
    `explainers` exists to show this statistic can tell the two apart.

    **The reference is load-bearing, not a default.** Deletion only removes
    evidence if the value written in carries no evidence, i.e. if the detector
    scores the reference as benign. Two natural-looking choices fail that test
    on this substrate and were measured before being rejected: the
    coordinate-wise median of the training set scores 0.99999 (malware) and the
    coordinate-wise median of the *benign* training set scores 0.71 (still
    malware), because a per-coordinate median of a 2,381-dimensional set is not
    a member of the distribution -- no real file takes the median value in every
    coordinate at once. Overwriting with either one deletes nothing, and F
    collapses to ~1e-5 for every explainer including the random control, which
    is how the fault was caught rather than published.

    Pass a bank of real, detector-verified-benign samples: F is averaged over
    the bank, so a single unrepresentative donor cannot set the result.

    Args:
        score_fn: maps a feature matrix to P(malware).
        X: alerts, shape (n, d).
        phi: attributions for those alerts, shape (n, d).
        reference: (d,) or a bank (R, d) of benign donors.
        steps: number of deletion points.

    Returns:
        F per alert, shape (n,), averaged over the reference bank.
    """
    n, d = X.shape
    bank = _as_reference_bank(reference, d)
    order = np.argsort(-np.abs(phi), axis=1)          # most attributed first
    base_scores = score_fn(X)
    predicted_malware = base_scores >= 0.5
    s0 = _decision_confidence(base_scores, predicted_malware)

    per_step = max(1, d // steps)
    rows_flat = np.arange(n)
    per_reference = np.zeros((len(bank), n), dtype=np.float64)

    for r, ref in enumerate(bank):
        drops = np.zeros((n, steps), dtype=np.float64)
        X_work = X.astype(np.float64, copy=True)
        for step in range(steps):
            lo, hi = step * per_step, min((step + 1) * per_step, d)
            if lo >= d:
                drops[:, step] = drops[:, step - 1]
                continue
            cols = order[:, lo:hi]
            rows = np.repeat(rows_flat, cols.shape[1])
            X_work[rows, cols.ravel()] = ref[cols.ravel()]
            s_j = _decision_confidence(score_fn(X_work), predicted_malware)
            drops[:, step] = np.clip((s0 - s_j) / np.maximum(s0, 0.5), 0.0, 1.0)
        per_reference[r] = drops.mean(axis=1)

    return per_reference.mean(axis=0)


def sufficiency_Phi(score_fn: ScoreFn,
                    X: np.ndarray,
                    phi: np.ndarray,
                    reference: np.ndarray,
                    top_k: int = 20) -> np.ndarray:
    """Phi: does the top-k of the explanation alone reproduce the decision?

    This is the *independent* criterion Equation (7) is tested against. It runs
    in the opposite direction to F -- keep the top-k and discard the rest --
    so a detector can score well on one and badly on the other, which is
    exactly what makes the calibration condition falsifiable.

    The reference carries the same requirement as in `faithfulness_F`: it must
    be scored benign, or every reduced input is malware whatever the explanation
    says and the sufficiency rate is 1.0 for the random control too. With a
    bank of donors the result is the fraction of donors for which the top-k
    alone reproduces the decision, which is a rate in [0, 1] rather than a bit.

    Returns:
        Shape (n,): fraction of reference donors under which the reduced input
        keeps the original decision. Binary when a single reference is given.
    """
    n, d = X.shape
    bank = _as_reference_bank(reference, d)
    order = np.argsort(-np.abs(phi), axis=1)[:, :top_k]
    base_scores = score_fn(X)
    predicted_malware = base_scores >= 0.5
    rows = np.repeat(np.arange(n), top_k)

    agree = np.zeros((len(bank), n), dtype=np.float64)
    for r, ref in enumerate(bank):
        X_reduced = np.broadcast_to(ref, (n, d)).astype(np.float64, copy=True)
        X_reduced[rows, order.ravel()] = X[rows, order.ravel()]
        reduced_malware = score_fn(X_reduced) >= 0.5
        agree[r] = reduced_malware == predicted_malware
    return agree.mean(axis=0)


def feature_group_index() -> np.ndarray:
    """Group id per feature column, shape (d,), indexing into GROUP_NAMES."""
    idx = np.full(sum(hi - lo for lo, hi in EMBER_GROUPS.values()), -1,
                  dtype=np.int64)
    for gid, name in enumerate(GROUP_NAMES):
        lo, hi = EMBER_GROUPS[name]
        idx[lo:hi] = gid
    if (idx < 0).any():
        raise ValueError("feature groups do not tile the vector")
    return idx


def disclosure_D(phi: np.ndarray, role: str, top_k: int = 20,
                 group_index: np.ndarray | None = None,
                 normalise: bool = True) -> np.ndarray:
    """Equation (10): D(z, R) = sum of w_R(c) over the components z reveals.

    comp(z) is the set of *feature groups* represented in the top-k attributed
    features -- the operational unit an analyst reads, rather than individual
    hashed columns.

    `normalise=True` divides by sum_c w_R(c), which the manuscript offers as the
    way to place the cost in [0, 1]. **That division is per-role, so normalised
    D is not comparable across roles**: if two roles' weight vectors differ by
    any positive scalar the normalised costs are identically equal, and for
    non-proportional weights the cross-role ordering is not preserved. Use
    `normalise=False` when comparing roles; keep normalisation only when
    comparing releases *within* one role against a tau_d in [0, 1].

    Returns:
        D per alert, shape (n,). In [0, 1] when normalised; in units of w_R
        otherwise.
    """
    weights = {g: ROLE_SENSITIVITY[g][role] for g in GROUP_NAMES}
    total = sum(weights.values())
    gidx = feature_group_index() if group_index is None else group_index

    order = np.argsort(-np.abs(phi), axis=1)[:, :top_k]
    revealed = gidx[order]                                 # (n, top_k)
    n = phi.shape[0]
    out = np.zeros(n, dtype=np.float64)
    weight_vec = np.array([weights[g] for g in GROUP_NAMES], dtype=np.float64)
    for i in range(n):
        out[i] = weight_vec[np.unique(revealed[i])].sum()
    if not normalise:
        return out
    if total <= 0:
        raise ValueError(f"role {role!r} has zero total sensitivity weight; "
                         "Equation (10)'s normalisation is undefined")
    return out / total


def is_admissible(F: np.ndarray, B: np.ndarray, D: np.ndarray,
                  tau_f: float, tau_b: float, tau_d: float,
                  A: np.ndarray | None = None,
                  tau_a: float | None = None) -> np.ndarray:
    """Definition 3, over the conditions this dataset can actually support.

    A_Gamma (actionability) needs a response playbook with preconditions and
    rollback. BODMAS has no basis for one, so `A` defaults to None and the
    admissibility reported is over {F, B, D} only. Passing a synthetic A and
    reporting the result as a four-condition admissibility rate would fabricate
    the governance layer the framework contributes; the caller must opt in.
    """
    ok = (F >= tau_f) & (B >= tau_b) & (D <= tau_d)
    if A is not None:
        if tau_a is None:
            raise ValueError("tau_a is required when A is supplied")
        ok &= (A >= tau_a)
    return ok


def summarise(name: str, values: np.ndarray) -> Mapping[str, float]:
    """Point estimate plus spread; never report a bare mean."""
    v = np.asarray(values, dtype=np.float64)
    return {
        f"{name}_mean": float(v.mean()),
        f"{name}_std": float(v.std()),
        f"{name}_min": float(v.min()),
        f"{name}_p25": float(np.percentile(v, 25)),
        f"{name}_median": float(np.median(v)),
        f"{name}_p75": float(np.percentile(v, 75)),
        f"{name}_max": float(v.max()),
    }


def deletion_curve(score_fn: ScoreFn,
                   X: np.ndarray,
                   phi: np.ndarray,
                   reference: np.ndarray,
                   k_values: Sequence[int] | None = None) -> dict:
    """Decision confidence after deleting the top-k features, k log-spaced.

    F on a linear schedule saturates on an easy detector: its first step deletes
    d/steps features, which is far more than the decision needs, so every
    competent explainer scores ~1.0 and the statistic stops separating them.
    A log-spaced schedule keeps the resolution where the action is.

    `flip_k` -- the smallest k at which the decision changes -- is the reportable
    quantity: it is in units an analyst understands (how many features had to be
    neutralised) rather than an area under an arbitrary curve.

    Returns:
        dict with the mean confidence per k, the per-alert flip_k (inf when the
        decision never flips within the largest k), and the flip rate.
    """
    n, d = X.shape
    ks = list(k_values) if k_values is not None else [
        k for k in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000) if k <= d]
    bank = _as_reference_bank(reference, d)
    order = np.argsort(-np.abs(phi), axis=1)
    base = score_fn(X)
    predicted_malware = base >= 0.5
    s0 = _decision_confidence(base, predicted_malware)

    conf = np.zeros((len(bank), len(ks), n), dtype=np.float64)
    for r, ref in enumerate(bank):
        for j, k in enumerate(ks):
            X_work = X.astype(np.float64, copy=True)
            cols = order[:, :k]
            rows = np.repeat(np.arange(n), k)
            X_work[rows, cols.ravel()] = ref[cols.ravel()]
            conf[r, j] = _decision_confidence(score_fn(X_work),
                                              predicted_malware)

    mean_conf = conf.mean(axis=0)                       # (len(ks), n)
    flipped = mean_conf < 0.5
    flip_k = np.full(n, np.inf)
    for j in range(len(ks) - 1, -1, -1):
        flip_k[flipped[j]] = ks[j]
    finite = flip_k[np.isfinite(flip_k)]
    return {
        "k_values": ks,
        "mean_confidence_by_k": [float(v) for v in mean_conf.mean(axis=1)],
        "baseline_confidence": float(s0.mean()),
        "flip_rate": float(np.isfinite(flip_k).mean()),
        "median_flip_k": float(np.median(finite)) if len(finite) else None,
        "mean_flip_k": float(finite.mean()) if len(finite) else None,
        "n": int(n),
    }
