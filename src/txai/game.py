"""E8: a finite instance of the explanation release game, and its equilibrium.

Proposition `sse` in the manuscript gives *existence* of a strong Stackelberg
equilibrium under compactness and continuity. It gives no procedure, and the
payoffs of Definition `game` carry free parameters. This module builds one
finite instance in which every payoff entry that *can* be measured is measured,
solves for the SSE, and reports how the equilibrium moves with the parameters
that cannot be.

**What is measured** (from the same substrate as E1-E7, treeshap, seed 42):

* ``q^{i,a}_sigma(delta)`` -- the joint law over (world, action), from the
  response kernel applied to the delivered view, exactly as in `release.py`.
* ``d_sigma(delta)`` -- Equation (10) disclosure cost of the *public* view.
* ``g_{m,sigma}(delta)`` -- attacker-visible content, as the fraction of the
  nine EMBER component groups the public view reveals.
* ``Pr[(z^D, omega) not|= Omega]`` -- instantiated as failure of the
  sufficiency test `Phi` on the delivered analyst view. This is an
  identification, not a theorem: the manuscript's `Omega` is any evidence
  obligation, and this instance picks the one the substrate can evaluate.

**What is stipulated** (declared here, swept in `sensitivity`):

* ``ell(a, H_i, c)`` -- the decision-loss matrix.
* ``kappa(m)``, ``beta(m)`` -- capability acquisition cost and the price the
  attacker puts on observed content.
* ``eta``, ``rho`` -- the defender's weights on evidence failure and disclosure.

**A degeneracy worth stating.** On BODMAS every alert in the substrate is
malware (E1: ``alert_malware_fraction = 1.0``), so ``q^{0,a} = 0`` and the
state-dependent loss collapses onto the malware column of ``ell``. That is a
consequence of F7 -- the corpus is separable enough that the alert population
carries no benign mass -- and not a property of the game. `PayoffTables`
records it so a reader is not left to infer it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .config import ACTIONS, GROUP_NAMES
from .metrics import disclosure_D, feature_group_index, sufficiency_Phi
from .release import ResponseKernel, joint_action_law

__all__ = [
    "GameInstance",
    "PayoffTables",
    "restrict_view",
    "attacker_visible_content",
    "build_payoffs",
    "solve_sse_pure",
    "solve_sse_milp",
    "solve_sse_multiple_lp",
]

logger = logging.getLogger(__name__)

# Index of the target action a* in ACTIONS. "dismiss" is the action an attacker
# steering an alert wants: it is the only response that takes no protective
# step at all.
TARGET_ACTION = "dismiss"


@dataclass(frozen=True)
class GameInstance:
    """The stipulated half of the instance. Everything here is a modelling choice.

    Attributes:
        loss: ell(a, H_i) as a mapping action -> (benign_loss, malware_loss).
        kappa: capability acquisition cost.
        beta: price the capability puts on observed content.
        eta: defender weight on evidence-obligation failure.
        rho: defender weight on disclosure cost.
        view_grid: the lattice of release widths the mechanism can be set to.
        public_ratio: the untrusted role's share of that width.
        role_defender: the role whose view drives the response kernel.
        role_attacker: the untrusted receiver role.
    """

    loss: Mapping[str, Tuple[float, float]] = field(default_factory=lambda: {
        # action        (benign world, malware world)
        "dismiss":      (0.00, 1.00),   # miss: the whole cost of the alert
        "monitor":      (0.05, 0.60),   # cheap, but does not contain
        "quarantine":   (0.40, 0.15),   # disruptive on a benign file
        "escalate":     (0.60, 0.05),   # analyst time is the dominant cost
    })
    kappa: Mapping[str, float] = field(default_factory=lambda: {
        "M4_observe": 0.00,      # observation of a released view costs nothing
        "M1_append": 0.02,       # appending bytes is the cheapest real edit
        "M1_imports": 0.05,      # adding imports needs a working loader path
        "M1_generic": 0.10,      # arbitrary bounded feature control is dearest
    })
    beta: Mapping[str, float] = field(default_factory=lambda: {
        "M4_observe": 1.00,      # disclosure *is* this capability's objective
        "M1_append": 0.30,
        "M1_imports": 0.30,
        "M1_generic": 0.30,
    })
    eta: float = 1.0
    rho: float = 1.0
    view_grid: Tuple[int, ...] = (5, 10, 20, 50, 100)
    public_ratio: float = 0.25
    role_defender: str = "analyst"
    role_attacker: str = "public"

    def public_width(self, k: int) -> int:
        """What the untrusted role sees under the same mechanism.

        `r` is one *role-indexed* mechanism, not one knob per role: widening it
        widens every role's view along the lattice, the untrusted one included.
        That coupling is what makes the game non-trivial. An instance in which
        the defender's own view is held fixed while only the public view varies
        has release as pure cost, and its equilibrium is "disclose nothing" at
        every parameter setting -- measured, and recorded in RESULTS.md.
        """
        return max(1, int(k * self.public_ratio))


@dataclass(frozen=True)
class PayoffTables:
    """The assembled bimatrix plus the measurements it was built from."""

    defender_labels: Tuple[str, ...]
    attacker_labels: Tuple[str, ...]
    U_D: np.ndarray                       # (n_sigma, n_k)
    U_A: np.ndarray                       # (n_sigma, n_k)
    measured: Mapping[str, object]

    def __post_init__(self) -> None:
        shape = (len(self.defender_labels), len(self.attacker_labels))
        for name, M in (("U_D", self.U_D), ("U_A", self.U_A)):
            if M.shape != shape:
                raise ValueError(f"{name} must be {shape}, got {M.shape}")
            if not np.isfinite(M).all():
                raise ValueError(f"{name} carries a non-finite payoff")


def restrict_view(phi: np.ndarray, top_k: int) -> np.ndarray:
    """The delivered view: the top-k attributed features, everything else zero.

    This is what makes the release mechanism a real strategy rather than a
    label. The response kernel and the disclosure witness both read the
    restricted array, so narrowing a release changes the action distribution and
    the cost together, which is the trade-off the game is about.
    """
    if top_k < 0:
        raise ValueError(f"top_k must be nonnegative, got {top_k}")
    out = np.zeros_like(phi)
    if top_k == 0:
        return out
    k = min(top_k, phi.shape[1])
    order = np.argsort(-np.abs(phi), axis=1)[:, :k]
    rows = np.repeat(np.arange(phi.shape[0]), k)
    out[rows, order.ravel()] = phi[rows, order.ravel()]
    return out


def attacker_visible_content(phi: np.ndarray, top_k: int,
                             group_index: np.ndarray | None = None) -> float:
    """G_m(z^A): the fraction of component groups the released view reveals.

    Deliberately *not* `disclosure_D`. D is the defender's cost and carries the
    role-sensitivity weights of the lattice; G is the attacker's value of what
    it can see, and weighting it by the defender's own sensitivity table would
    assume the two agree. An unweighted group count assumes only that a
    capability gains from learning which components drive the decision.
    """
    if top_k == 0:
        return 0.0
    gidx = feature_group_index() if group_index is None else group_index
    k = min(top_k, phi.shape[1])
    order = np.argsort(-np.abs(phi), axis=1)[:, :k]
    revealed = gidx[order]
    counts = np.array([len(np.unique(row)) for row in revealed], dtype=np.float64)
    return float(counts.mean() / len(GROUP_NAMES))


def build_payoffs(instance: GameInstance,
                  kernels: Sequence[ResponseKernel],
                  attacker_moves: Sequence[Tuple[str, str, np.ndarray]],
                  y_alerts: np.ndarray,
                  score_fn,
                  X_by_move: Mapping[str, np.ndarray],
                  reference: np.ndarray) -> PayoffTables:
    """Assemble U_D and U_A over (release level x kernel) against (m, delta).

    Args:
        instance: the stipulated parameters.
        kernels: the response kernels available to the defender.
        attacker_moves: (label, capability, phi_delta) per attacker strategy,
            where phi_delta are the attributions under that transformation.
        y_alerts: latent world per alert.
        score_fn: the detector's score function, for the evidence obligation.
        X_by_move: perturbed alert matrix per attacker-move label.
        reference: benign donor bank for the sufficiency test.

    Returns:
        PayoffTables, with the per-move measurements retained.
    """
    gidx = feature_group_index()
    target = ACTIONS.index(TARGET_ACTION)
    loss_matrix = np.array([[instance.loss[a][0], instance.loss[a][1]]
                            for a in ACTIONS], dtype=np.float64).T   # (2, |U|)

    # The evidence obligation is evaluated on the delivered view, so it depends
    # on the release width as well as on the transformation. This is the term
    # that gives the defender a reason to release at all.
    evidence_failure: Dict[Tuple[int, str], float] = {}
    for k in instance.view_grid:
        for label, _cap, phi_d in attacker_moves:
            passes = sufficiency_Phi(score_fn, X_by_move[label], phi_d,
                                     reference, top_k=k)
            evidence_failure[(k, label)] = float(1.0 - passes.mean())

    sigma_labels: List[str] = []
    rows_D: List[List[float]] = []
    rows_A: List[List[float]] = []
    detail: List[dict] = []

    for k in instance.view_grid:
        k_a = instance.public_width(k)
        for kernel in kernels:
            sigma_labels.append(f"k={k},pi={kernel.name}")
            row_D, row_A = [], []
            for label, cap, phi_d in attacker_moves:
                z_d = restrict_view(phi_d, k)
                z_a = restrict_view(phi_d, k_a)

                pi_dist = kernel.apply(z_d, gidx)
                q = joint_action_law(pi_dist, y_alerts)          # (2, |U|)
                decision_loss = float((q * loss_matrix).sum())
                q_target = float(q[:, target].sum())

                d_cost = float(
                    disclosure_D(z_a, instance.role_attacker, top_k=k_a,
                                 group_index=gidx, normalise=True).mean())
                gain = attacker_visible_content(phi_d, k_a, gidx)
                fail = evidence_failure[(k, label)]

                u_d = (-decision_loss - instance.eta * fail
                       - instance.rho * d_cost)
                u_a = (q_target
                       + instance.beta[cap] * gain
                       - instance.kappa[cap])
                row_D.append(u_d)
                row_A.append(u_a)
                detail.append({
                    "sigma": sigma_labels[-1], "move": label, "capability": cap,
                    "release_width": k, "public_width": k_a,
                    "decision_loss": decision_loss, "q_target": q_target,
                    "disclosure_D": d_cost, "attacker_gain": gain,
                    "evidence_failure": fail,
                    "U_D": u_d, "U_A": u_a,
                })
            rows_D.append(row_D)
            rows_A.append(row_A)

    world_mass = np.array([float((y_alerts == 0).mean()),
                           float((y_alerts == 1).mean())])
    return PayoffTables(
        defender_labels=tuple(sigma_labels),
        attacker_labels=tuple(label for label, _c, _p in attacker_moves),
        U_D=np.array(rows_D, dtype=np.float64),
        U_A=np.array(rows_A, dtype=np.float64),
        measured={
            "world_mass": {"benign": world_mass[0], "malware": world_mass[1]},
            "loss_column_collapsed": bool(world_mass[0] == 0.0),
            "evidence_failure": {f"k={k}|{label}": v
                                 for (k, label), v in evidence_failure.items()},
            "cells": detail,
        })


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def _best_responses(u_a_row: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Indices attaining the attacker's maximum, within tolerance."""
    return np.flatnonzero(u_a_row >= u_a_row.max() - tol)


def solve_sse_pure(tables: PayoffTables) -> dict:
    """SSE when the defender is restricted to a pure strategy: enumeration.

    Included because it is the honest baseline. If the mixed solution below
    does no better, the MILP bought nothing and the paper should say so.
    """
    best = None
    for i in range(tables.U_D.shape[0]):
        br = _best_responses(tables.U_A[i])
        j = int(br[np.argmax(tables.U_D[i, br])])   # optimistic tie-breaking
        value = float(tables.U_D[i, j])
        if best is None or value > best["defender_utility"]:
            best = {"defender_strategy": tables.defender_labels[i],
                    "attacker_strategy": tables.attacker_labels[j],
                    "defender_utility": value,
                    "attacker_utility": float(tables.U_A[i, j]),
                    "n_tied_best_responses": int(len(br))}
    assert best is not None
    return best


def solve_sse_milp(tables: PayoffTables) -> dict:
    """Strong Stackelberg equilibrium over mixed defender strategies (DOBSS).

    Variables are ``z[i, j]`` (defender mass on sigma_i, conditioned on the
    attacker playing j), the binary follower selection ``q[j]``, and the
    follower's equilibrium value ``a``. Exactly one ``q[j]`` is one, so
    ``x_i = sum_j z[i, j]`` is the defender's mixed strategy and the objective
    is linear. Optimistic tie-breaking is automatic: the maximisation is free to
    pick the best-for-the-defender member of the best-response set.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    n_i, n_j = tables.U_D.shape
    n_z = n_i * n_j
    n_var = n_z + n_j + 1                      # z, q, a
    big_m = float(np.ptp(tables.U_A) + 1.0)

    def z_idx(i: int, j: int) -> int:
        return i * n_j + j

    c = np.zeros(n_var)
    c[:n_z] = -tables.U_D.ravel()              # milp minimises

    rows: List[np.ndarray] = []
    lb: List[float] = []
    ub: List[float] = []

    # sum_ij z_ij = 1
    r = np.zeros(n_var); r[:n_z] = 1.0
    rows.append(r); lb.append(1.0); ub.append(1.0)

    # sum_j q_j = 1
    r = np.zeros(n_var); r[n_z:n_z + n_j] = 1.0
    rows.append(r); lb.append(1.0); ub.append(1.0)

    # sum_i z_ij - q_j <= 0  : no defender mass on an unselected follower column
    for j in range(n_j):
        r = np.zeros(n_var)
        for i in range(n_i):
            r[z_idx(i, j)] = 1.0
        r[n_z + j] = -1.0
        rows.append(r); lb.append(-np.inf); ub.append(0.0)

    # follower optimality: 0 <= a - sum_i U_A[i,j] x_i <= (1 - q_j) M
    for j in range(n_j):
        r = np.zeros(n_var)
        for i in range(n_i):
            for jj in range(n_j):
                r[z_idx(i, jj)] = -tables.U_A[i, j]     # x_i = sum_jj z_i,jj
        r[-1] = 1.0                                     # + a
        rows.append(r); lb.append(0.0); ub.append(np.inf)

        r2 = r.copy()
        r2[n_z + j] = big_m                             # + M q_j
        rows.append(r2); lb.append(-np.inf); ub.append(big_m)

    A = np.vstack(rows)
    constraints = LinearConstraint(A, np.array(lb), np.array(ub))

    integrality = np.zeros(n_var)
    integrality[n_z:n_z + n_j] = 1
    var_lb = np.zeros(n_var); var_lb[-1] = -np.inf
    var_ub = np.ones(n_var); var_ub[-1] = np.inf

    res = milp(c=c, constraints=constraints, integrality=integrality,
               bounds=Bounds(var_lb, var_ub))
    if not res.success:
        raise RuntimeError(f"MILP did not solve: {res.message}")

    z = res.x[:n_z].reshape(n_i, n_j)
    x = z.sum(axis=1)
    j_star = int(np.argmax(res.x[n_z:n_z + n_j]))
    support = {tables.defender_labels[i]: float(x[i])
               for i in range(n_i) if x[i] > 1e-7}
    return {
        "defender_mixed_strategy": support,
        "attacker_strategy": tables.attacker_labels[j_star],
        "defender_utility": float(-res.fun),
        "attacker_utility": float(tables.U_A[:, j_star] @ x),
        "support_size": len(support),
        "status": res.message,
    }


def solve_sse_multiple_lp(tables: PayoffTables) -> dict:
    """Independent cross-check: the multiple-LP method of Conitzer and Sandholm.

    Solves one LP per attacker pure strategy, forcing that strategy to be a best
    response, and keeps the best. It shares no code path with the MILP above,
    so agreement between the two is evidence that the formulation is right and
    not merely that it is self-consistent.
    """
    from scipy.optimize import linprog

    n_i, n_j = tables.U_D.shape
    best = None
    for j in range(n_j):
        # maximise U_D[:, j] . x  s.t.  (U_A[:, j] - U_A[:, jj]) . x >= 0
        A_ub = np.array([tables.U_A[:, jj] - tables.U_A[:, j]
                         for jj in range(n_j) if jj != j], dtype=np.float64)
        res = linprog(c=-tables.U_D[:, j],
                      A_ub=A_ub if len(A_ub) else None,
                      b_ub=np.zeros(len(A_ub)) if len(A_ub) else None,
                      A_eq=np.ones((1, n_i)), b_eq=np.array([1.0]),
                      bounds=[(0.0, 1.0)] * n_i, method="highs")
        if not res.success:
            continue
        value = float(-res.fun)
        if best is None or value > best["defender_utility"]:
            best = {
                "defender_mixed_strategy": {
                    tables.defender_labels[i]: float(res.x[i])
                    for i in range(n_i) if res.x[i] > 1e-7},
                "attacker_strategy": tables.attacker_labels[j],
                "defender_utility": value,
                "attacker_utility": float(tables.U_A[:, j] @ res.x),
            }
    if best is None:
        raise RuntimeError("no attacker pure strategy admitted a feasible LP")
    return best
