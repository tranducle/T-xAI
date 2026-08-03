"""BODMAS loading, layout verification, and temporal splitting.

The corpus is not redistributed with this repository. Point `TXAI_DATA_DIR` at a
directory holding `bodmas.npz` and `bodmas_metadata.csv` as published by the
BODMAS authors (see README, "Data"), or pass `dataset_dir=` explicitly. Access is
read-only; nothing here writes into the data directory.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .config import EMBER_GROUPS, N_FEATURES

__all__ = ["Dataset", "load_bodmas", "temporal_split", "verify_feature_layout"]

logger = logging.getLogger(__name__)

#: Repository root, three levels up from ``src/txai/data.py``.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Where the corpora live. Override with the ``TXAI_DATA_DIR`` environment
#: variable; the default keeps a clone self-contained without hardcoding any
#: machine-specific path.
DATASET_DIR = Path(os.environ.get("TXAI_DATA_DIR", REPO_ROOT / "data"))


@dataclass(frozen=True)
class Dataset:
    """Feature matrix, labels, and the parsed acquisition timestamps."""

    X: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray  # datetime64[ns], NaT where unparsed

    def __post_init__(self) -> None:
        if not (len(self.X) == len(self.y) == len(self.timestamps)):
            raise ValueError(
                f"length mismatch: X={len(self.X)} y={len(self.y)} "
                f"ts={len(self.timestamps)}")


def verify_feature_layout(X: np.ndarray, n_check: int = 3000) -> dict:
    """Confirm the EMBER-v2 column order by arithmetic invariant, not by faith.

    ByteHistogram and ByteEntropyHistogram are both normalised by their own sum
    in `ember/features.py`, so columns 0:256 and 256:512 must each sum to
    exactly 1 in every row. No other 256-wide block in the vector does that, so
    the two checks together pin the offset of everything downstream.

    Raises:
        ValueError: when the layout does not hold, which means the dataset or
            the extractor version is not the one this code assumes.
    """
    if X.shape[1] != N_FEATURES:
        raise ValueError(f"expected {N_FEATURES} features, got {X.shape[1]}")

    sample = X[:n_check]
    report = {}
    for name in ("byte_histogram", "byte_entropy"):
        lo, hi = EMBER_GROUPS[name]
        rowsum = sample[:, lo:hi].sum(axis=1)
        report[name] = {"mean": float(rowsum.mean()), "max_dev":
                        float(np.abs(rowsum - 1.0).max())}
        if report[name]["max_dev"] > 1e-4:
            raise ValueError(
                f"{name} rows do not sum to 1 (max deviation "
                f"{report[name]['max_dev']:.2e}); the EMBER-v2 column order "
                "assumed in config.EMBER_GROUPS does not hold for this file")

    covered = sum(hi - lo for lo, hi in EMBER_GROUPS.values())
    if covered != N_FEATURES:
        raise ValueError(f"groups cover {covered} of {N_FEATURES} columns")

    logger.info("EMBER-v2 layout verified on %d rows", len(sample))
    return report


def load_bodmas(dataset_dir: Path | None = None) -> Dataset:
    """Load BODMAS features, labels and timestamps, verifying the layout."""
    root = dataset_dir or DATASET_DIR
    npz_path = root / "bodmas.npz"
    meta_path = root / "bodmas_metadata.csv"
    try:
        arrays = np.load(npz_path)
        X, y = arrays["X"], arrays["y"]
    except FileNotFoundError:
        logger.error("BODMAS archive not found at %s", npz_path)
        raise

    verify_feature_layout(X)

    meta = pd.read_csv(meta_path)
    if len(meta) != len(X):
        raise ValueError(
            f"metadata has {len(meta)} rows but X has {len(X)}; the row "
            "correspondence this code relies on cannot be assumed")
    timestamps = pd.to_datetime(meta["timestamp"], errors="coerce",
                                utc=True).values

    logger.info("loaded BODMAS: X=%s, %d benign / %d malware",
                X.shape, int((y == 0).sum()), int((y == 1).sum()))
    return Dataset(X=X, y=y, timestamps=timestamps)


def temporal_split(data: Dataset, test_fraction: float = 0.2
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Split by acquisition time: train on the past, test on the future.

    Rows with an unparsed timestamp sort last under numpy's NaT ordering, which
    would quietly place them all in the test set. They are dropped instead, and
    the count is logged.

    Returns:
        (train_index, test_index) into the dataset's rows.
    """
    valid = np.flatnonzero(~np.isnat(data.timestamps))
    dropped = len(data.timestamps) - len(valid)
    if dropped:
        logger.warning("dropping %d rows with an unparsable timestamp", dropped)

    order = valid[np.argsort(data.timestamps[valid], kind="stable")]
    cut = int((1.0 - test_fraction) * len(order))
    train_idx, test_idx = order[:cut], order[cut:]
    logger.info("temporal split at %s: train=%d test=%d",
                data.timestamps[order[cut]], len(train_idx), len(test_idx))
    return train_idx, test_idx
