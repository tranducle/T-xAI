"""Release layer: the response kernel, Adv^TV, and the Theorem 1 bridge check.

The response kernel is built so that its Lipschitz constant is **exact, not
estimated**. Theorem 1's hypothesis (Eq. `lipschitz`) asks for

    TV( pi(.|z), pi(.|z') ) <= L_pi * d_Z(z, z').

Take pi(.|z) = G @ g(z), where g(z) aggregates the normalised attribution mass
into feature groups and G is column-stochastic over the action set. Both maps
are column-stochastic, and a column-stochastic matrix is an l1 non-expansion:

    ||W v||_1 = sum_a |sum_j W_aj v_j|
             <= sum_a sum_j W_aj |v_j|
              = sum_j |v_j| * (sum_a W_aj) = ||v||_1.

Hence L_pi = 1 exactly. That matters: an estimated L_pi would make the measured
slack in the bound partly an artefact of the estimate, and the tightness number
would mean nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .config import ACTIONS, GROUP_NAMES, ROLE_ORDER
from .metrics import (attribution_distance, disclosure_D, feature_group_index,
                      normalise_attribution)

__all__ = [
    "ResponseKernel",
    "joint_action_law",
    "adv_tv",
    "bridge_check",
    "role_disclosure_profile",
    "check_view_monotonicity",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResponseKernel:
    """pi(a | z) = G g(z): a column-stochastic map from components to actions.

    Attributes:
        G: shape (n_actions, n_groups), every column summing to 1.
        name: label recorded with the results.
    """

    G: np.ndarray
    name: str = "default"

    def __post_init__(self) -> None:
        if self.G.shape != (len(ACTIONS), len(GROUP_NAMES)):
            raise ValueError(
                f"G must be {(len(ACTIONS), len(GROUP_NAMES))}, got {self.G.shape}")
        col_sums = self.G.sum(axis=0)
        if not np.allclose(col_sums, 1.0, atol=1e-9):
            raise ValueError(
                "G must be column-stochastic; L_pi = 1 is what makes the "
                f"Theorem 1 comparison exact. Column sums: {col_sums}")
        if (self.G < 0).any():
            raise ValueError("G must be non-negative")

    @property
    def lipschitz_constant(self) -> float:
        """L_pi = 1, by the l1 non-expansiveness of a column-stochastic matrix."""
        return 1.0

    def apply(self, phi: np.ndarray,
              group_index: np.ndarray | None = None) -> np.ndarray:
        """Action distribution per alert, shape (n, n_actions)."""
        gidx = feature_group_index() if group_index is None else group_index
        p = normalise_attribution(phi)                       # (n, d)
        n_groups = len(GROUP_NAMES)
        g = np.zeros((p.shape[0], n_groups), dtype=np.float64)
        np.add.at(g.T, gidx, p.T)                            # sum mass per group
        return g @ self.G.T                                  # (n, n_actions)


def sharp_kernel(seed: int = 0, temperature: float = 1.0) -> ResponseKernel:
    """A kernel whose columns concentrate on one action per component group.

    `temperature` interpolates between one-hot columns (0 -> sharp) and uniform
    columns (large -> diffuse). Used for the kernel ablation: a diffuse kernel
    contracts harder, so the Theorem 1 bound should get looser, and if the
    measurement does not show that, the implementation is wrong.
    """
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(len(ACTIONS), len(GROUP_NAMES)))
    if temperature <= 0:
        G = np.zeros_like(logits)
        G[logits.argmax(axis=0), np.arange(len(GROUP_NAMES))] = 1.0
        return ResponseKernel(G=G, name="sharp")
    e = np.exp(logits / temperature)
    return ResponseKernel(G=e / e.sum(axis=0, keepdims=True),
                          name=f"softmax_t{temperature:g}")


def joint_action_law(pi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """q[i, a] = P(world = i) * E[ pi(a | z) | world = i ], shape (2, n_actions).

    This is the joint law over (latent world, defender action) that
    Proposition `advbound` and Theorem 1 are stated in terms of.
    """
    q = np.zeros((2, pi.shape[1]), dtype=np.float64)
    n = len(y)
    for world in (0, 1):
        mask = y == world
        if not mask.any():
            continue
        q[world] = (mask.sum() / n) * pi[mask].mean(axis=0)
    return q


def adv_tv(pi_clean: np.ndarray, pi_perturbed: Sequence[np.ndarray],
           y: np.ndarray) -> dict:
    """Adv^TV = sup over the perturbation set of TV between joint laws.

    Returns:
        dict with the supremum, the per-perturbation values, and the argmax.
    """
    q0 = joint_action_law(pi_clean, y)
    values = []
    for pi_d in pi_perturbed:
        q = joint_action_law(pi_d, y)
        values.append(0.5 * np.abs(q - q0).sum())
    values_arr = np.asarray(values, dtype=np.float64)
    return {
        "adv_tv": float(values_arr.max()),
        "per_perturbation": [float(v) for v in values_arr],
        "argmax_perturbation": int(values_arr.argmax()),
        "mean": float(values_arr.mean()),
    }


def bridge_check(phi_clean: np.ndarray,
                 phi_perturbed: Sequence[np.ndarray],
                 y: np.ndarray,
                 kernel: ResponseKernel) -> dict:
    """Measure both sides of Theorem 1 and report the slack.

    LHS: Adv^TV over the same perturbation set.
    RHS: min{1, L_pi * E_x[1 - B^z(x)]}.

    The theorem is proved, so a violation here means this code is wrong, not
    that the theorem is. `holds` is therefore an assertion about the
    implementation and is checked on every run.
    """
    gidx = feature_group_index()
    pi_clean_dist = kernel.apply(phi_clean, gidx)
    pi_perturbed_dists = [kernel.apply(p, gidx) for p in phi_perturbed]

    lhs = adv_tv(pi_clean_dist, pi_perturbed_dists, y)

    worst = np.zeros(len(phi_clean), dtype=np.float64)
    for phi_d in phi_perturbed:
        worst = np.maximum(worst, attribution_distance(phi_clean, phi_d))
    expected_instability = float(worst.mean())          # E[1 - B^z]
    rhs = min(1.0, kernel.lipschitz_constant * expected_instability)

    holds = lhs["adv_tv"] <= rhs + 1e-9
    if not holds:
        logger.error("Theorem 1 appears violated (LHS=%.6f > RHS=%.6f). The "
                     "theorem is proved, so this is an implementation fault.",
                     lhs["adv_tv"], rhs)

    return {
        "lhs_adv_tv": lhs["adv_tv"],
        "rhs_bound": rhs,
        "L_pi": kernel.lipschitz_constant,
        "expected_instability": expected_instability,
        "mean_B": 1.0 - expected_instability,
        "slack_absolute": rhs - lhs["adv_tv"],
        "tightness_ratio": (lhs["adv_tv"] / rhs) if rhs > 0 else float("nan"),
        "holds": bool(holds),
        "kernel": kernel.name,
        "per_perturbation_tv": lhs["per_perturbation"],
    }


def role_disclosure_profile(phi: np.ndarray, top_k: int = 20) -> dict:
    """Cross-role behaviour of Equation (10), raw and normalised.

    Proposition `lp` claims monotonicity under *component inclusion* at each
    fixed role -- see `check_view_monotonicity` -- and never claims that D falls
    as privilege rises. This function measures the cross-role direction anyway,
    because the manuscript's normalisation remark has a consequence worth
    recording:

    * **raw** D orders across roles whenever w_R is pointwise ordered, since it
      is then a sum of pointwise-ordered nonnegative terms. `raw_monotone`
      therefore validates the weight table in `config`; a violation means that
      table contradicts the privilege lattice it claims to encode.
    * **normalised** D generally does *not* order across roles. Dividing by
      sum_c w_R(c) is a per-role rescaling: two roles whose weights differ by a
      positive scalar receive identically equal normalised cost, and
      non-proportional weights can reorder. `normalised_monotone` is measured,
      not asserted -- it is expected to be False, and that is the finding.
    """
    raw = {r: disclosure_D(phi, r, top_k, normalise=False) for r in ROLE_ORDER}
    norm = {r: disclosure_D(phi, r, top_k, normalise=True) for r in ROLE_ORDER}

    def _violations(values: dict) -> tuple[int, list]:
        total, pairs = 0, []
        for lo_role, hi_role in zip(ROLE_ORDER, ROLE_ORDER[1:]):
            bad = int((values[lo_role] < values[hi_role] - 1e-12).sum())
            total += bad
            pairs.append({"pair": f"{lo_role}>={hi_role}", "violations": bad})
        return total, pairs

    raw_v, raw_pairs = _violations(raw)
    norm_v, norm_pairs = _violations(norm)
    return {
        "n_alerts": int(phi.shape[0]),
        "raw_monotone": raw_v == 0,
        "raw_violations": raw_v,
        "raw_pairs": raw_pairs,
        "mean_raw_D_by_role": {r: float(v.mean()) for r, v in raw.items()},
        "normalised_monotone": norm_v == 0,
        "normalised_violations": norm_v,
        "normalised_pairs": norm_pairs,
        "mean_normalised_D_by_role": {r: float(v.mean()) for r, v in norm.items()},
    }


def check_view_monotonicity(phi: np.ndarray,
                            top_k_grid: Sequence[int]) -> dict:
    """Least privilege in the view direction: a narrower release discloses no more.

    D counts distinct component groups, so widening the released view can only
    add groups. A decrease would mean `disclosure_D` is not monotone in top_k
    and the least-privilege reading of the lattice would not hold numerically.
    """
    ks = sorted(top_k_grid)
    values = {k: disclosure_D(phi, "analyst", k) for k in ks}
    violations = 0
    for k_small, k_big in zip(ks, ks[1:]):
        violations += int((values[k_small] > values[k_big] + 1e-12).sum())
    return {
        "monotone": violations == 0,
        "total_violations": violations,
        "mean_D_by_top_k": {int(k): float(v.mean()) for k, v in values.items()},
    }
