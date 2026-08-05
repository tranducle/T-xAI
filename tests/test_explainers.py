"""Tests for the two control explainers and for explainer statefulness.

The controls are what license every claim of the form "the metric measures the
explanation". A control whose behaviour depends on how the calling code happens
to construct it silently voids that licence, which is exactly what happened in
the first full run: `pipeline.explain` rebuilt the explainer on every call, the
seeded RNG replayed the same draws, and the instability control was recorded
with B = 1.000 at every perturbation family and budget. These tests pin the
semantics so the artefact cannot come back unnoticed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from txai_exp.explainers import build_explainer  # noqa: E402
from txai_exp.metrics import attribution_distance, robustness_B  # noqa: E402

D = 64


def _rows(n: int = 8, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, D))


# --- the instability control -----------------------------------------------

def test_random_control_gives_unrelated_attributions_on_successive_calls() -> None:
    explainer = build_explainer("random", n_features=D, seed=42)
    X = _rows()
    first, second = explainer.explain(X), explainer.explain(X)
    assert not np.array_equal(first, second)


def test_random_control_scores_far_below_perfect_robustness() -> None:
    """The point of the control: B near 1 must be earned, not automatic."""
    explainer = build_explainer("random", n_features=D, seed=42)
    X = _rows(64)
    clean = explainer.explain(X)
    perturbed = [explainer.explain(X) for _ in range(5)]
    B = robustness_B(clean, perturbed)
    assert B.mean() < 0.8, f"random control scored B={B.mean():.3f}"


def test_random_control_is_reproducible_from_the_seed_and_call_order() -> None:
    X = _rows()
    a = build_explainer("random", n_features=D, seed=7)
    b = build_explainer("random", n_features=D, seed=7)
    assert np.array_equal(a.explain(X), b.explain(X))
    assert np.array_equal(a.explain(X), b.explain(X))       # streams stay in step


# --- the gaming control ----------------------------------------------------

def test_constant_control_ignores_its_input_entirely() -> None:
    explainer = build_explainer("constant", n_features=D, seed=42)
    X = _rows()
    out = explainer.explain(X)
    assert np.array_equal(out[0], out[-1])                  # same for every row
    assert np.array_equal(out, explainer.explain(X + 10.0))  # and every input


def test_constant_control_attains_B_exactly_one() -> None:
    """Robustness alone is maximised by an explanation that explains nothing."""
    explainer = build_explainer("constant", n_features=D, seed=42)
    X = _rows(32)
    clean = explainer.explain(X)
    perturbed = [explainer.explain(X + delta) for delta in (0.1, 1.0, 10.0)]
    B = robustness_B(clean, perturbed)
    assert np.allclose(B, 1.0)
    assert np.allclose(attribution_distance(clean, perturbed[0]), 0.0)


# --- the regression test for the artefact ----------------------------------

def test_pipeline_explain_does_not_reset_a_stateful_explainer() -> None:
    """Reproduces the 2026-08-02 bug at the layer where it actually occurred."""
    from types import SimpleNamespace

    from txai_exp import pipeline

    sub = SimpleNamespace(model=None, background=None, score_fn=None, seed=42)
    cfg = SimpleNamespace(lime_samples=10)
    X = _rows()
    first = pipeline.explain(sub, cfg, "random", X)
    second = pipeline.explain(sub, cfg, "random", X)
    assert not np.array_equal(first, second), (
        "pipeline.explain rebuilt the explainer and replayed the same draws")
