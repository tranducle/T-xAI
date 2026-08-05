"""Known-answer tests for the scoring layer.

Every test below pins a value that can be computed by hand, or a structural
property the framework's proofs depend on. The point is not coverage: it is
that a silent sign flip, a wrong axis, or an off-by-one in the feature layout
would change a published number, and none of those announce themselves at
runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from txai_exp import metrics, release  # noqa: E402
from txai_exp.config import ACTIONS, EMBER_GROUPS, GROUP_NAMES, ROLE_SENSITIVITY  # noqa: E402
from txai_exp.perturbations import apply_perturbation, check_invariants  # noqa: E402


# --------------------------------------------------------------------------
# normalise_attribution
# --------------------------------------------------------------------------

def test_normalisation_produces_a_distribution_and_discards_sign() -> None:
    phi = np.array([[-3.0, 1.0, 0.0, 0.0]])
    p = metrics.normalise_attribution(phi)
    np.testing.assert_allclose(p, [[0.75, 0.25, 0.0, 0.0]])
    assert p.sum() == pytest.approx(1.0)


def test_an_all_zero_explanation_maps_to_uniform_not_to_a_point_mass() -> None:
    """"Says nothing" must not be encoded as "says one specific thing"."""
    p = metrics.normalise_attribution(np.zeros((1, 4)))
    np.testing.assert_allclose(p, [[0.25, 0.25, 0.25, 0.25]])


# --------------------------------------------------------------------------
# d_Z and B (Equation 6)
# --------------------------------------------------------------------------

def test_distance_is_zero_for_identical_and_one_for_disjoint_support() -> None:
    a = np.array([[1.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.0, 5.0]])
    assert metrics.attribution_distance(a, a)[0] == pytest.approx(0.0)
    assert metrics.attribution_distance(a, b)[0] == pytest.approx(1.0)


def test_distance_matches_a_hand_computed_total_variation() -> None:
    # p = (0.5, 0.5, 0), q = (0.25, 0.25, 0.5) -> TV = 0.5*(0.25+0.25+0.5) = 0.5
    a = np.array([[1.0, 1.0, 0.0]])
    b = np.array([[1.0, 1.0, 2.0]])
    assert metrics.attribution_distance(a, b)[0] == pytest.approx(0.5)


def test_distance_is_scale_invariant() -> None:
    """B must not change because an explainer rescales its output."""
    a = np.array([[1.0, 2.0, 3.0]])
    assert metrics.attribution_distance(a, 100.0 * a)[0] == pytest.approx(0.0)


def test_B_is_one_under_no_change_and_zero_under_a_disjoint_flip() -> None:
    clean = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert metrics.robustness_B(clean, [clean.copy()]) == pytest.approx([1.0, 1.0])
    flipped = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    assert metrics.robustness_B(clean, [flipped]) == pytest.approx([0.0, 0.0])


def test_B_takes_the_worst_case_over_the_perturbation_set() -> None:
    """Equation (6) is a supremum; averaging would overstate robustness."""
    clean = np.array([[1.0, 0.0]])
    mild = np.array([[0.9, 0.1]])       # d_Z = 0.1
    severe = np.array([[0.0, 1.0]])     # d_Z = 1.0
    assert metrics.robustness_B(clean, [mild, severe])[0] == pytest.approx(0.0)
    assert metrics.robustness_B(clean, [severe, mild])[0] == pytest.approx(0.0)


def test_B_refuses_an_empty_perturbation_set() -> None:
    with pytest.raises(ValueError, match="at least one perturbation"):
        metrics.robustness_B(np.zeros((2, 3)), [])


# --------------------------------------------------------------------------
# F and Phi -- and the fact that they are different tests
# --------------------------------------------------------------------------

def _one_feature_model(active: int = 0):
    """P(malware) determined entirely by feature `active`."""
    def score(X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-12.0 * (np.asarray(X)[:, active] - 0.5)))
    return score


def test_faithfulness_rewards_the_explanation_that_names_the_real_driver() -> None:
    score = _one_feature_model(active=0)
    X = np.tile(np.array([[0.95, 0.5, 0.5, 0.5, 0.5]]), (4, 1))
    reference = np.zeros(5)

    truthful = np.tile(np.array([[10.0, 0.0, 0.0, 0.0, 0.0]]), (4, 1))
    misleading = np.tile(np.array([[0.0, 10.0, 1.0, 1.0, 1.0]]), (4, 1))

    f_true = metrics.faithfulness_F(score, X, truthful, reference, steps=5)
    f_false = metrics.faithfulness_F(score, X, misleading, reference, steps=5)
    assert (f_true > f_false).all(), (f_true, f_false)
    assert (f_true >= 0).all() and (f_true <= 1).all()


def test_sufficiency_is_one_only_when_the_kept_features_carry_the_decision() -> None:
    score = _one_feature_model(active=0)
    X = np.tile(np.array([[0.95, 0.5, 0.5, 0.5, 0.5]]), (2, 1))
    reference = np.zeros(5)

    truthful = np.tile(np.array([[10.0, 0.0, 0.0, 0.0, 0.0]]), (2, 1))
    misleading = np.tile(np.array([[0.0, 10.0, 0.0, 0.0, 0.0]]), (2, 1))

    assert metrics.sufficiency_Phi(score, X, truthful, reference, top_k=1).tolist() == [1, 1]
    assert metrics.sufficiency_Phi(score, X, misleading, reference, top_k=1).tolist() == [0, 0]


def test_F_and_Phi_can_disagree_which_is_what_makes_Eq7_falsifiable() -> None:
    """If one implied the other by construction, E3 would be vacuous.

    Feature 1 is a near-perfect proxy for the driver in this model, so deleting
    it moves the score (high F) while keeping it alone does not reproduce the
    decision (Phi = 0).
    """
    def score(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return 1.0 / (1.0 + np.exp(-12.0 * (0.5 * X[:, 0] + 0.5 * X[:, 1] - 0.5)))

    X = np.tile(np.array([[0.9, 0.9, 0.0]]), (3, 1))
    reference = np.zeros(3)
    phi = np.tile(np.array([[0.0, 10.0, 0.0]]), (3, 1))   # points only at feature 1

    F = metrics.faithfulness_F(score, X, phi, reference, steps=3)
    Phi = metrics.sufficiency_Phi(score, X, phi, reference, top_k=1)
    assert (F > 0).all()
    assert Phi.tolist() == [0, 0, 0]


def test_faithfulness_handles_more_steps_than_features() -> None:
    score = _one_feature_model(active=0)
    X = np.tile(np.array([[0.9, 0.5, 0.5]]), (2, 1))
    F = metrics.faithfulness_F(score, X, np.ones((2, 3)), np.zeros(3), steps=20)
    assert F.shape == (2,) and np.isfinite(F).all()


# --------------------------------------------------------------------------
# Feature layout and disclosure (Equation 10)
# --------------------------------------------------------------------------

def test_feature_groups_tile_the_vector_exactly_once() -> None:
    gidx = metrics.feature_group_index()
    assert gidx.shape == (2381,)
    assert set(np.unique(gidx)) == set(range(len(GROUP_NAMES)))
    bounds = sorted(EMBER_GROUPS.values())
    for (_, hi), (lo_next, _) in zip(bounds, bounds[1:]):
        assert hi == lo_next, "groups must be contiguous with no gap or overlap"


def test_disclosure_equals_the_hand_summed_weight_of_the_revealed_groups() -> None:
    phi = np.zeros((1, 2381))
    phi[0, 0] = 5.0                       # byte_histogram
    phi[0, 1000] = 4.0                    # imports
    D = metrics.disclosure_D(phi, "analyst", top_k=2)[0]

    total = sum(ROLE_SENSITIVITY[g]["analyst"] for g in GROUP_NAMES)
    expected = (ROLE_SENSITIVITY["byte_histogram"]["analyst"]
                + ROLE_SENSITIVITY["imports"]["analyst"]) / total
    assert D == pytest.approx(expected)


def test_disclosure_counts_a_group_once_however_many_features_it_contributes() -> None:
    phi = np.zeros((1, 2381))
    phi[0, :10] = 5.0                     # ten features, all byte_histogram
    D = metrics.disclosure_D(phi, "public", top_k=10)[0]
    total = sum(ROLE_SENSITIVITY[g]["public"] for g in GROUP_NAMES)
    assert D == pytest.approx(ROLE_SENSITIVITY["byte_histogram"]["public"] / total)


def test_raw_disclosure_falls_as_privilege_rises() -> None:
    """Pointwise-ordered weights order the raw Eq. (10) sum. Validates config."""
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(50, 2381))
    result = release.role_disclosure_profile(phi, top_k=20)
    assert result["raw_monotone"], result["raw_pairs"]


def test_normalising_Eq10_destroys_cross_role_comparability() -> None:
    """A measured caveat, not a bug: the [0,1] rescaling is per-role.

    Equation (10) offers division by sum_c w_R(c) to place the cost in [0, 1].
    That divisor depends on R, so the normalised costs of two roles are not on
    a common scale and their ordering need not survive. This test records the
    failure so no downstream table compares normalised D across roles.
    """
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(50, 2381))
    result = release.role_disclosure_profile(phi, top_k=20)
    assert result["raw_monotone"]
    assert not result["normalised_monotone"], (
        "normalised D happened to order across roles here; the caveat recorded "
        "in RESULTS must be re-derived before it is reported")


def test_proportional_weights_make_normalised_disclosure_role_blind() -> None:
    """The mechanism behind the caveat, isolated: scaling cancels exactly."""
    base = {g: ROLE_SENSITIVITY[g]["public"] for g in GROUP_NAMES}
    scaled = {g: 0.01 * w for g, w in base.items()}
    revealed = ("byte_histogram", "imports", "sections")

    d_base = sum(base[g] for g in revealed) / sum(base.values())
    d_scaled = sum(scaled[g] for g in revealed) / sum(scaled.values())
    assert d_base == pytest.approx(d_scaled), (
        "a role whose weights are a positive multiple of another's receives an "
        "identical normalised cost, so normalised D cannot separate them")


def test_a_narrower_release_never_discloses_more() -> None:
    rng = np.random.default_rng(1)
    phi = rng.normal(size=(50, 2381))
    result = release.check_view_monotonicity(phi, [5, 10, 20, 50])
    assert result["monotone"], result


# --------------------------------------------------------------------------
# The response kernel: L_pi = 1 is what makes Theorem 1 measurable
# --------------------------------------------------------------------------

def test_kernel_rejects_a_matrix_that_is_not_column_stochastic() -> None:
    bad = np.ones((len(ACTIONS), len(GROUP_NAMES)))
    with pytest.raises(ValueError, match="column-stochastic"):
        release.ResponseKernel(G=bad)


def test_kernel_output_is_a_distribution_over_actions() -> None:
    kernel = release.sharp_kernel(seed=0, temperature=1.0)
    rng = np.random.default_rng(2)
    pi = kernel.apply(rng.normal(size=(20, 2381)))
    assert pi.shape == (20, len(ACTIONS))
    np.testing.assert_allclose(pi.sum(axis=1), 1.0, atol=1e-9)


def test_kernel_is_an_l1_non_expansion_so_L_pi_is_exactly_one() -> None:
    """The single numerical fact Theorem 1's RHS rests on.

    If this fails, the measured bound is meaningless because L_pi would have to
    be estimated rather than known.
    """
    kernel = release.sharp_kernel(seed=3, temperature=0.7)
    rng = np.random.default_rng(4)
    a = rng.normal(size=(200, 2381))
    b = rng.normal(size=(200, 2381))

    d_explanation = metrics.attribution_distance(a, b)
    pi_a, pi_b = kernel.apply(a), kernel.apply(b)
    d_action = 0.5 * np.abs(pi_a - pi_b).sum(axis=1)

    assert (d_action <= d_explanation + 1e-9).all()
    assert kernel.lipschitz_constant == 1.0


def test_joint_action_law_is_a_probability_distribution() -> None:
    kernel = release.sharp_kernel(seed=5)
    rng = np.random.default_rng(6)
    pi = kernel.apply(rng.normal(size=(100, 2381)))
    y = rng.integers(0, 2, size=100)
    q = release.joint_action_law(pi, y)
    assert q.shape == (2, len(ACTIONS))
    assert q.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Theorem 1
# --------------------------------------------------------------------------

def test_the_bridge_bound_holds_on_random_explanations() -> None:
    kernel = release.sharp_kernel(seed=7, temperature=1.0)
    rng = np.random.default_rng(8)
    clean = rng.normal(size=(200, 2381))
    perturbed = [clean + 0.5 * rng.normal(size=clean.shape) for _ in range(5)]
    y = rng.integers(0, 2, size=200)

    result = release.bridge_check(clean, perturbed, y, kernel)
    assert result["holds"], result
    assert 0.0 <= result["lhs_adv_tv"] <= result["rhs_bound"] + 1e-9


def test_the_bridge_is_exactly_tight_when_nothing_is_perturbed() -> None:
    kernel = release.sharp_kernel(seed=9)
    rng = np.random.default_rng(10)
    clean = rng.normal(size=(50, 2381))
    y = rng.integers(0, 2, size=50)
    result = release.bridge_check(clean, [clean.copy()], y, kernel)
    assert result["lhs_adv_tv"] == pytest.approx(0.0, abs=1e-12)
    assert result["rhs_bound"] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# Perturbations stay inside the feature space they are allowed to move in
# --------------------------------------------------------------------------

def _synthetic_batch(n: int = 32) -> np.ndarray:
    rng = np.random.default_rng(11)
    X = np.abs(rng.normal(size=(n, 2381))) + 0.01
    for group in ("byte_histogram", "byte_entropy"):
        lo, hi = EMBER_GROUPS[group]
        X[:, lo:hi] /= X[:, lo:hi].sum(axis=1, keepdims=True)
    X[:, 616] = rng.integers(10_000, 5_000_000, size=n)     # file size
    return X


@pytest.mark.parametrize("name", ["append_bytes", "add_imports", "generic"])
def test_every_perturbation_preserves_the_feature_space_invariants(name: str) -> None:
    X = _synthetic_batch()
    rng = np.random.default_rng(12)
    out = apply_perturbation(name, X, rng, strength=0.05)
    check_invariants(X, out)          # raises on violation
    assert out.shape == X.shape
    assert not np.allclose(out, X), f"{name} did not change anything"


def test_append_bytes_moves_the_histogram_toward_the_appended_content() -> None:
    """The convex-combination update is the physical content of this family."""
    X = _synthetic_batch(n=4)
    rng = np.random.default_rng(13)
    weak = apply_perturbation("append_bytes", X, rng, strength=0.001)
    rng = np.random.default_rng(13)
    strong = apply_perturbation("append_bytes", X, rng, strength=0.5)

    lo, hi = EMBER_GROUPS["byte_histogram"]
    d_weak = np.abs(weak[:, lo:hi] - X[:, lo:hi]).sum(axis=1)
    d_strong = np.abs(strong[:, lo:hi] - X[:, lo:hi]).sum(axis=1)
    assert (d_strong > d_weak).all()


def test_a_perturbation_that_shrinks_the_file_is_rejected() -> None:
    X = _synthetic_batch(n=4)
    shrunk = X.copy()
    shrunk[:, 616] *= 0.5
    with pytest.raises(ValueError, match="append-only"):
        check_invariants(X, shrunk)


def test_a_perturbation_that_breaks_histogram_normalisation_is_rejected() -> None:
    X = _synthetic_batch(n=4)
    broken = X.copy()
    broken[:, 0] += 0.5
    with pytest.raises(ValueError, match="no longer sums to 1"):
        check_invariants(X, broken)


def test_unknown_perturbation_names_fail_loudly() -> None:
    with pytest.raises(ValueError, match="unknown perturbation"):
        apply_perturbation("teleport", _synthetic_batch(4),
                           np.random.default_rng(0), 0.1)


# --------------------------------------------------------------------------
# Admissibility (Definition 3)
# --------------------------------------------------------------------------

def test_admissibility_requires_every_condition_simultaneously() -> None:
    F = np.array([0.9, 0.9, 0.9, 0.1])
    B = np.array([0.9, 0.9, 0.1, 0.9])
    D = np.array([0.1, 0.9, 0.1, 0.1])
    ok = metrics.is_admissible(F, B, D, tau_f=0.5, tau_b=0.5, tau_d=0.5)
    assert ok.tolist() == [True, False, False, False]


def test_supplying_actionability_without_its_threshold_is_an_error() -> None:
    """A_Gamma has no basis in this dataset; using it must be a deliberate act."""
    with pytest.raises(ValueError, match="tau_a"):
        metrics.is_admissible(np.array([1.0]), np.array([1.0]), np.array([0.0]),
                              tau_f=0.5, tau_b=0.5, tau_d=0.5,
                              A=np.array([1.0]))


# --------------------------------------------------------------------------
# The deletion reference -- the choice that silently voided the first run
# --------------------------------------------------------------------------

def test_a_reference_bank_averages_over_donors() -> None:
    score = _one_feature_model(active=0)
    X = np.tile(np.array([[0.95, 0.5, 0.5]]), (2, 1))
    phi = np.tile(np.array([[10.0, 0.0, 0.0]]), (2, 1))

    benign = np.zeros(3)                      # score(benign) ~ 0.002
    useless = np.array([0.95, 0.5, 0.5])      # identical to the alert

    f_benign = metrics.faithfulness_F(score, X, phi, benign, steps=3)
    f_useless = metrics.faithfulness_F(score, X, phi, useless, steps=3)
    f_bank = metrics.faithfulness_F(
        score, X, phi, np.vstack([benign, useless]), steps=3)

    assert f_bank == pytest.approx(0.5 * (f_benign + f_useless))
    assert (f_useless < f_benign).all(), (
        "a reference equal to the input deletes nothing; if this were not "
        "detectable the reference choice could not be validated")


def test_sufficiency_over_a_bank_is_a_rate_not_a_bit() -> None:
    score = _one_feature_model(active=0)
    X = np.array([[0.95, 0.5, 0.5]])
    phi = np.array([[10.0, 0.0, 0.0]])
    bank = np.vstack([np.zeros(3), np.array([0.95, 0.5, 0.5])])
    out = metrics.sufficiency_Phi(score, X, phi, bank, top_k=1)
    assert out.shape == (1,)
    assert 0.0 <= out[0] <= 1.0


def test_a_reference_of_the_wrong_width_is_rejected() -> None:
    score = _one_feature_model(active=0)
    with pytest.raises(ValueError, match="reference must be"):
        metrics.faithfulness_F(score, np.zeros((2, 3)), np.ones((2, 3)),
                               np.zeros(7), steps=2)


def test_the_reference_bank_refuses_donors_the_detector_calls_malware() -> None:
    """The guard that would have caught the median-reference fault up front."""
    from txai_exp.pipeline import build_reference_bank

    X = np.arange(40, dtype=np.float64).reshape(20, 2)
    y = np.zeros(20, dtype=int)
    y[10:] = 1

    with pytest.raises(ValueError, match="scored below"):
        build_reference_bank(X, y, lambda Z: np.full(len(Z), 0.99), seed=0)

    bank = build_reference_bank(X, y, lambda Z: np.zeros(len(Z)), seed=0, size=3)
    assert bank.shape == (3, 2)
    assert all((row == X[:10]).all(axis=1).any() for row in bank), \
        "donors must be real training rows, not synthesised points"


def test_the_deletion_curve_separates_explainers_that_F_cannot() -> None:
    """A hard-threshold model: F saturates for anything sane, flip_k does not."""
    def score(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return (X[:, :3].sum(axis=1) > 1.5).astype(float)

    X = np.tile(np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]), (4, 1))
    benign = np.zeros(6)

    truthful = np.tile(np.array([[9.0, 8.0, 7.0, 0.0, 0.0, 0.0]]), (4, 1))
    misleading = np.tile(np.array([[0.0, 0.0, 0.0, 9.0, 8.0, 7.0]]), (4, 1))

    good = metrics.deletion_curve(score, X, truthful, benign, k_values=[1, 2, 3, 6])
    bad = metrics.deletion_curve(score, X, misleading, benign, k_values=[1, 2, 3, 6])

    assert good["median_flip_k"] < bad["median_flip_k"], (good, bad)
    assert good["flip_rate"] == 1.0


def test_the_deletion_curve_reports_no_flip_rather_than_inventing_one() -> None:
    def score(X: np.ndarray) -> np.ndarray:
        return np.full(len(np.asarray(X)), 0.99)      # never changes its mind

    out = metrics.deletion_curve(score, np.ones((3, 6)), np.ones((3, 6)),
                                 np.zeros(6), k_values=[1, 3, 6])
    assert out["flip_rate"] == 0.0
    assert out["median_flip_k"] is None
