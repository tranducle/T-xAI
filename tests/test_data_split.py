"""Tests for the data layer.

The split is the one place where a mistake inflates every downstream number at
once and leaves no trace in any result file, so it is pinned here rather than
trusted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from txai_exp.config import EMBER_GROUPS, N_FEATURES  # noqa: E402
from txai_exp.data import Dataset, temporal_split, verify_feature_layout  # noqa: E402


def _dataset(n: int = 100, n_bad: int = 0) -> Dataset:
    stamps = pd.to_datetime(
        ["2019-01-01"] * 0 + [f"2019-{1 + i % 12:02d}-{1 + i % 28:02d}"
                              for i in range(n)], utc=True).values
    if n_bad:
        stamps = stamps.copy()
        stamps[:n_bad] = np.datetime64("NaT")
    return Dataset(X=np.zeros((n, 3)), y=np.zeros(n, dtype=int), timestamps=stamps)


def test_no_training_sample_is_newer_than_any_test_sample() -> None:
    """The definition of a temporal split; anything else is leakage."""
    data = _dataset(200)
    train, test = temporal_split(data, test_fraction=0.2)
    assert data.timestamps[train].max() <= data.timestamps[test].min()


def test_the_split_partitions_the_rows_with_no_overlap() -> None:
    data = _dataset(200)
    train, test = temporal_split(data, test_fraction=0.25)
    assert set(train).isdisjoint(test)
    assert len(train) + len(test) == 200
    assert len(test) == pytest.approx(50, abs=1)


def test_unparsable_timestamps_are_dropped_not_dumped_into_the_test_set() -> None:
    """NaT sorts last in numpy, so the naive split would put all of them in test."""
    data = _dataset(100, n_bad=10)
    train, test = temporal_split(data, test_fraction=0.2)
    assert len(train) + len(test) == 90
    kept = np.concatenate([train, test])
    assert not np.isnat(data.timestamps[kept]).any()


def test_dataset_rejects_misaligned_arrays() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        Dataset(X=np.zeros((5, 3)), y=np.zeros(4), timestamps=np.zeros(5, "datetime64[ns]"))


def test_layout_verification_rejects_a_matrix_of_the_wrong_width() -> None:
    with pytest.raises(ValueError, match="expected 2381 features"):
        verify_feature_layout(np.zeros((10, 100)))


def test_layout_verification_rejects_unnormalised_histograms() -> None:
    """The check that licenses every group-level disclosure result."""
    X = np.zeros((10, N_FEATURES))
    X[:, 0:256] = 1.0 / 256.0
    X[:, 256:512] = 1.0 / 256.0
    verify_feature_layout(X)                      # passes

    X_bad = X.copy()
    X_bad[:, 0] += 0.5
    with pytest.raises(ValueError, match="do not sum to 1"):
        verify_feature_layout(X_bad)


def test_declared_groups_cover_the_vector_exactly() -> None:
    assert sum(hi - lo for lo, hi in EMBER_GROUPS.values()) == N_FEATURES
