"""E8: the release-game instance and the two equilibrium solvers.

The solver tests use hand-computable bimatrices rather than the BODMAS payoffs,
so a failure here is a fault in the formulation and not in the measurement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from txai_exp.game import (PayoffTables, attacker_visible_content,  # noqa: E402
                           restrict_view, solve_sse_milp,
                           solve_sse_multiple_lp, solve_sse_pure)


def _tables(U_D: np.ndarray, U_A: np.ndarray) -> PayoffTables:
    return PayoffTables(
        defender_labels=tuple(f"s{i}" for i in range(U_D.shape[0])),
        attacker_labels=tuple(f"k{j}" for j in range(U_D.shape[1])),
        U_D=np.asarray(U_D, dtype=np.float64),
        U_A=np.asarray(U_A, dtype=np.float64),
        measured={})


# ---------------------------------------------------------------------------
# View restriction
# ---------------------------------------------------------------------------

def test_restrict_view_keeps_exactly_the_top_k_by_magnitude():
    phi = np.array([[0.1, -0.9, 0.4, 0.0]])
    out = restrict_view(phi, 2)
    assert out.tolist() == [[0.0, -0.9, 0.4, 0.0]]


def test_restrict_view_zero_releases_nothing():
    phi = np.array([[0.1, -0.9, 0.4]])
    assert not restrict_view(phi, 0).any()


def test_restrict_view_beyond_width_is_the_identity():
    phi = np.array([[0.1, -0.9, 0.4]])
    assert np.array_equal(restrict_view(phi, 99), phi)


def test_restrict_view_rejects_a_negative_width():
    with pytest.raises(ValueError):
        restrict_view(np.zeros((1, 3)), -1)


def test_attacker_gain_is_zero_when_nothing_is_released():
    assert attacker_visible_content(np.random.default_rng(0).normal(
        size=(4, 2381)), 0) == 0.0


def test_attacker_gain_rises_with_the_release_width():
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(8, 2381))
    narrow = attacker_visible_content(phi, 2)
    wide = attacker_visible_content(phi, 50)
    assert 0.0 < narrow <= wide <= 1.0


# ---------------------------------------------------------------------------
# Payoff table invariants
# ---------------------------------------------------------------------------

def test_payoff_tables_reject_a_shape_mismatch():
    with pytest.raises(ValueError):
        PayoffTables(("a",), ("x", "y"), np.zeros((1, 2)), np.zeros((1, 1)), {})


def test_payoff_tables_reject_a_non_finite_entry():
    with pytest.raises(ValueError):
        PayoffTables(("a",), ("x",), np.array([[np.nan]]), np.array([[0.0]]), {})


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def test_pure_sse_breaks_ties_in_the_defenders_favour():
    # Row 0: the attacker is indifferent between both columns; the strong
    # equilibrium takes the 5, not the -5.
    U_A = np.array([[1.0, 1.0]])
    U_D = np.array([[-5.0, 5.0]])
    out = solve_sse_pure(_tables(U_D, U_A))
    assert out["defender_utility"] == 5.0
    assert out["n_tied_best_responses"] == 2


def test_pure_sse_anticipates_the_best_response():
    # Row 0 looks better to the defender, but the attacker will not play the
    # column that makes it so; row 1 is the correct commitment.
    U_A = np.array([[0.0, 1.0],
                    [1.0, 0.0]])
    U_D = np.array([[9.0, 0.0],
                    [3.0, 1.0]])
    out = solve_sse_pure(_tables(U_D, U_A))
    assert out["defender_strategy"] == "s1"
    assert out["defender_utility"] == 3.0


def test_milp_matches_the_multiple_lp_method_on_random_games():
    rng = np.random.default_rng(7)
    for _ in range(12):
        U_D = rng.normal(size=(5, 4))
        U_A = rng.normal(size=(5, 4))
        t = _tables(U_D, U_A)
        assert solve_sse_milp(t)["defender_utility"] == pytest.approx(
            solve_sse_multiple_lp(t)["defender_utility"], abs=1e-7)


def test_milp_is_never_worse_than_the_pure_optimum():
    rng = np.random.default_rng(11)
    for _ in range(12):
        t = _tables(rng.normal(size=(6, 5)), rng.normal(size=(6, 5)))
        assert (solve_sse_milp(t)["defender_utility"]
                >= solve_sse_pure(t)["defender_utility"] - 1e-9)


def test_mixing_can_strictly_beat_every_pure_commitment():
    # The textbook Stackelberg gap: committing to 50/50 makes the attacker
    # switch to the column the defender prefers, which no pure row achieves.
    U_D = np.array([[2.0, 4.0],
                    [1.0, 3.0]])
    U_A = np.array([[1.0, 0.0],
                    [0.0, 1.0]])
    t = _tables(U_D, U_A)
    mixed = solve_sse_milp(t)["defender_utility"]
    assert mixed > solve_sse_pure(t)["defender_utility"] + 1e-6
    assert mixed == pytest.approx(solve_sse_multiple_lp(t)["defender_utility"],
                                  abs=1e-7)


def test_a_dominant_attacker_column_fixes_the_best_response():
    U_A = np.array([[0.0, 5.0],
                    [0.0, 5.0]])
    U_D = np.array([[9.0, 1.0],
                    [8.0, 2.0]])
    out = solve_sse_milp(_tables(U_D, U_A))
    assert out["attacker_strategy"] == "k1"
    assert out["defender_utility"] == pytest.approx(2.0, abs=1e-7)
