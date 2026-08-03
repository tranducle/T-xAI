"""Explanation methods behind one interface, including a random control.

The random control is not filler. Every metric in `metrics` claims to measure a
property *of the explanation*. If a uniformly random attribution vector scores
the same as SHAP, the metric is measuring the detector or the perturbation, not
the explanation, and every result built on it is void. The control is the only
thing that can distinguish those cases, so it is a first-class explainer here.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Protocol

import numpy as np

__all__ = ["EXPLAINER_FACTORY", "register_explainer", "build_explainer",
           "Explainer"]

logger = logging.getLogger(__name__)


class Explainer(Protocol):
    def explain(self, X: np.ndarray) -> np.ndarray:
        """Attributions for each row, shape (n, d)."""


EXPLAINER_FACTORY: Dict[str, Callable[..., Explainer]] = {}


def register_explainer(name: str) -> Callable[..., Callable[..., Explainer]]:
    def decorator(fn):
        EXPLAINER_FACTORY[name] = fn
        return fn
    return decorator


class _TreeShapExplainer:
    """Exact Shapley values for tree ensembles.

    `shap` returns either an (n, d) array or an (n, d, n_classes) array
    depending on the model wrapper. Both shapes are handled explicitly: taking
    the wrong slice silently yields the *benign*-class attributions, which are
    the negation of what every metric here assumes.
    """

    def __init__(self, model) -> None:
        import shap
        self._explainer = shap.TreeExplainer(model)

    def explain(self, X: np.ndarray) -> np.ndarray:
        values = self._explainer.shap_values(np.asarray(X, dtype=np.float32))
        if isinstance(values, list):          # legacy per-class list
            values = values[1] if len(values) > 1 else values[0]
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 3:                  # (n, d, n_classes)
            values = values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]
        if values.ndim != 2:
            raise ValueError(f"unexpected SHAP output shape {values.shape}")
        return values


class _LimeExplainer:
    """Local surrogate attributions.

    `discretize_continuous=False` is deliberate: LIME's default quartile
    discretisation is meaningless for the 1,280 hashed import bins, which are
    mostly zero, and would collapse them into a single bucket.
    """

    def __init__(self, background: np.ndarray, score_fn, seed: int,
                 num_samples: int = 2000, num_features: int | None = None) -> None:
        from lime.lime_tabular import LimeTabularExplainer
        self._score_fn = score_fn
        self._num_samples = num_samples
        self._d = background.shape[1]
        self._num_features = num_features or self._d
        self._explainer = LimeTabularExplainer(
            np.asarray(background, dtype=np.float64),
            mode="classification", discretize_continuous=False,
            random_state=seed)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        p1 = self._score_fn(X)
        return np.column_stack([1.0 - p1, p1])

    def explain(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros((len(X), self._d), dtype=np.float64)
        for i, row in enumerate(np.asarray(X, dtype=np.float64)):
            exp = self._explainer.explain_instance(
                row, self._predict, labels=(1,),
                num_features=self._num_features,
                num_samples=self._num_samples)
            for feature_idx, weight in exp.as_map()[1]:
                out[i, feature_idx] = weight
        return out


class _RandomExplainer:
    """Fresh random attributions on every call -- the *instability* control.

    The RNG advances per call, so explaining the same rows twice gives
    unrelated vectors. That is deliberate: it is the lower end of the
    robustness scale, and it is what makes a high measured B mean something.

    Reproducibility comes from the seed plus a fixed call order, not from
    per-call determinism. Callers must therefore reuse one instance for a whole
    block -- rebuilding it resets the stream and silently turns this control
    into the constant one. `pipeline.explain` caches instances for exactly this
    reason; the 2026-08-02 first run did not, and recorded B = 1.000 for this
    control at every family and budget.
    """

    def __init__(self, n_features: int, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._d = n_features

    def explain(self, X: np.ndarray) -> np.ndarray:
        return self._rng.normal(size=(len(X), self._d))


class _ConstantExplainer:
    """One fixed vector for every row and every call -- the *gaming* control.

    An explanation that ignores its input is perfectly stable, so it attains
    B = 1 exactly and makes the Theorem 1 bound hold with 0 <= 0. It is the
    cheapest way to satisfy a robustness threshold, and including it is the
    only way the write-up can say that robustness alone is not evidence of
    anything -- admissibility has to conjoin F.
    """

    def __init__(self, n_features: int, seed: int) -> None:
        self._w = np.random.default_rng(seed).normal(size=n_features)

    def explain(self, X: np.ndarray) -> np.ndarray:
        return np.tile(self._w, (len(X), 1))


@register_explainer("treeshap")
def _make_treeshap(model=None, **_) -> Explainer:
    return _TreeShapExplainer(model)


@register_explainer("lime")
def _make_lime(background=None, score_fn=None, seed: int = 0,
               lime_samples: int = 2000, **_) -> Explainer:
    return _LimeExplainer(background, score_fn, seed, num_samples=lime_samples)


@register_explainer("random")
def _make_random(n_features: int = 2381, seed: int = 0, **_) -> Explainer:
    return _RandomExplainer(n_features, seed)


@register_explainer("constant")
def _make_constant(n_features: int = 2381, seed: int = 0, **_) -> Explainer:
    return _ConstantExplainer(n_features, seed)


def build_explainer(name: str, **kwargs) -> Explainer:
    try:
        return EXPLAINER_FACTORY[name](**kwargs)
    except KeyError:
        raise ValueError(f"unknown explainer {name!r}; "
                         f"registered: {sorted(EXPLAINER_FACTORY)}") from None
