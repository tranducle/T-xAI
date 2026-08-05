"""Threat class M_1: input-only, functionality-preserving perturbations.

These are **feature-space proxies for append-only binary edits**, not binary
rewrites. Nothing here produces a PE file, so no claim may be made about
evasion of a real pipeline. What they do preserve is the structural property
that makes an edit functionality-preserving: appended content only ever *adds*
-- bytes, sections, imports -- and never removes or relocates what the binary
already needs.

Two arithmetic invariants are enforced on every perturbed batch, because a
perturbation that broke them would silently move samples off the data manifold
and every downstream robustness number would be measuring an artefact:

1. the byte and byte-entropy histograms still sum to exactly 1;
2. the file size never shrinks.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict

import numpy as np

from .config import EMBER_GROUPS, SIZE_FEATURE_INDEX

__all__ = ["PERTURBATION_FACTORY", "register_perturbation", "apply_perturbation",
           "check_invariants"]

logger = logging.getLogger(__name__)

PerturbFn = Callable[[np.ndarray, np.random.Generator, float], np.ndarray]
PERTURBATION_FACTORY: Dict[str, PerturbFn] = {}

_IMPORTS_COUNT_INDEX = SIZE_FEATURE_INDEX + 4   # GeneralFileInfo[4] = imports
_VSIZE_INDEX = SIZE_FEATURE_INDEX + 1           # GeneralFileInfo[1] = vsize


def register_perturbation(name: str) -> Callable[[PerturbFn], PerturbFn]:
    def decorator(fn: PerturbFn) -> PerturbFn:
        PERTURBATION_FACTORY[name] = fn
        return fn
    return decorator


@register_perturbation("append_bytes")
def append_bytes(X: np.ndarray, rng: np.random.Generator,
                 strength: float) -> np.ndarray:
    """Append `strength * size` bytes drawn from a random content distribution.

    Appending k bytes with byte distribution q to a file of n bytes takes the
    normalised histogram h to (n*h + k*q)/(n + k). That is exact for the byte
    histogram. The byte-entropy histogram is mixed the same way, which is an
    approximation -- its true update depends on the window structure of the
    appended region, which feature vectors do not carry. Recorded as an
    approximation rather than presented as exact.
    """
    out = X.astype(np.float64, copy=True)
    n_rows = len(out)
    size = np.maximum(out[:, SIZE_FEATURE_INDEX], 1.0)
    k = np.maximum(strength * size, 1.0)
    w = (k / (size + k))[:, None]                    # mixing weight of new content

    for group in ("byte_histogram", "byte_entropy"):
        lo, hi = EMBER_GROUPS[group]
        q = rng.dirichlet(np.ones(hi - lo), size=n_rows)
        out[:, lo:hi] = (1.0 - w) * out[:, lo:hi] + w * q

    out[:, SIZE_FEATURE_INDEX] = size + k
    out[:, _VSIZE_INDEX] = np.maximum(out[:, _VSIZE_INDEX], out[:, _VSIZE_INDEX] + k)
    return out


@register_perturbation("add_imports")
def add_imports(X: np.ndarray, rng: np.random.Generator,
                strength: float) -> np.ndarray:
    """Declare additional imported functions the binary never calls.

    Import features are hashed count bins, so adding an import increments a
    bin. Unused imports do not change behaviour, which is what makes this
    functionality-preserving.
    """
    out = X.astype(np.float64, copy=True)
    lo, hi = EMBER_GROUPS["imports"]
    width = hi - lo
    n_touch = max(1, int(round(strength * width)))
    rows = np.arange(len(out))
    for _ in range(n_touch):
        cols = rng.integers(lo, hi, size=len(out))
        out[rows, cols] += 1.0
    out[:, _IMPORTS_COUNT_INDEX] += n_touch
    return out


@register_perturbation("generic")
def generic_bounded(X: np.ndarray, rng: np.random.Generator,
                    strength: float) -> np.ndarray:
    """Unstructured non-negative nudge on a random subset of coordinates.

    Deliberately *not* physically grounded. It is the control: if robustness
    under this looks the same as under the two structured families, the
    structure in those families is not doing any work and the framing of
    Delta_z as functionality-preserving adds nothing to the measurement.
    Histogram blocks are renormalised so the invariants still hold.
    """
    out = X.astype(np.float64, copy=True)
    mask = rng.random(out.shape) < strength
    out += rng.random(out.shape) * mask * np.abs(out).mean(axis=0, keepdims=True)
    for group in ("byte_histogram", "byte_entropy"):
        lo, hi = EMBER_GROUPS[group]
        block = np.maximum(out[:, lo:hi], 0.0)
        total = block.sum(axis=1, keepdims=True)
        out[:, lo:hi] = np.where(total > 0, block / total, out[:, lo:hi])
    out[:, SIZE_FEATURE_INDEX] = np.maximum(out[:, SIZE_FEATURE_INDEX],
                                            X[:, SIZE_FEATURE_INDEX])
    return out


def check_invariants(X_original: np.ndarray, X_perturbed: np.ndarray,
                     atol: float = 1e-6) -> None:
    """Raise if a perturbation left the feature space it is allowed to move in."""
    for group in ("byte_histogram", "byte_entropy"):
        lo, hi = EMBER_GROUPS[group]
        rowsum = X_perturbed[:, lo:hi].sum(axis=1)
        dev = float(np.abs(rowsum - 1.0).max())
        if dev > atol:
            raise ValueError(
                f"{group} no longer sums to 1 after perturbation "
                f"(max deviation {dev:.2e}); the perturbed samples are off the "
                "feature manifold and any robustness score would be an artefact")
    shrunk = int((X_perturbed[:, SIZE_FEATURE_INDEX]
                  < X_original[:, SIZE_FEATURE_INDEX] - atol).sum())
    if shrunk:
        raise ValueError(f"{shrunk} samples shrank; append-only was violated")


def apply_perturbation(name: str, X: np.ndarray, rng: np.random.Generator,
                       strength: float, verify: bool = True) -> np.ndarray:
    """Apply a registered perturbation and verify it stayed admissible."""
    try:
        fn = PERTURBATION_FACTORY[name]
    except KeyError:
        raise ValueError(
            f"unknown perturbation {name!r}; "
            f"registered: {sorted(PERTURBATION_FACTORY)}") from None
    out = fn(X, rng, strength)
    if verify:
        check_invariants(X, out)
    return out
