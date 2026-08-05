"""Detector substrate. The detector is not a contribution here -- it is the
thing being explained -- so the registry exists to vary it in the ablation, not
to compete on accuracy."""

from __future__ import annotations

import logging
from typing import Callable, Dict, Protocol

import numpy as np

__all__ = ["DETECTOR_FACTORY", "register_detector", "build_detector",
           "score_function"]

logger = logging.getLogger(__name__)


class Detector(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray): ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


DETECTOR_FACTORY: Dict[str, Callable[[int], Detector]] = {}


def register_detector(name: str) -> Callable[..., Callable[[int], Detector]]:
    def decorator(fn):
        DETECTOR_FACTORY[name] = fn
        return fn
    return decorator


@register_detector("histgb")
def _histgb(seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                          random_state=seed)


@register_detector("xgboost")
def _xgboost(seed: int):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                         tree_method="hist", random_state=seed,
                         n_jobs=-1, eval_metric="logloss")


@register_detector("randomforest")
def _randomforest(seed: int):
    """Depth-capped on purpose.

    Exact TreeSHAP costs O(trees x leaves x depth^2). An unbounded forest on
    107k samples grows ~10^4 leaves per tree, which puts a single explanation
    batch into the hours. The cap keeps the ablation affordable while still
    supplying a genuinely different inductive bias (bagged full-feature trees
    rather than boosted histograms). Recorded because it is a deviation from
    the library default, not a tuned choice.
    """
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1,
                                  random_state=seed)


def build_detector(name: str, seed: int) -> Detector:
    try:
        return DETECTOR_FACTORY[name](seed)
    except KeyError:
        raise ValueError(f"unknown detector {name!r}; "
                         f"registered: {sorted(DETECTOR_FACTORY)}") from None


def score_function(model: Detector) -> Callable[[np.ndarray], np.ndarray]:
    """P(malware) as a plain callable, so metrics never touch the model API.

    Casts to float32 because the tree libraries were fitted on float32 and
    scoring a float64 view of the same values otherwise costs a full copy on
    every deletion step -- of which there are `steps` per alert batch.
    """
    def score(X: np.ndarray) -> np.ndarray:
        return model.predict_proba(np.asarray(X, dtype=np.float32))[:, 1]
    return score
