"""Build the E1-E8 substrate on EMBER-2018 instead of BODMAS.

Why this module exists at all: on BODMAS the detector is separable enough that
*every* alert it raises is malware (E1: ``alert_malware_fraction = 1.0``). The
latent world ``H`` in Definition `game` is then a constant, the joint law over
(world, action) equals its own action marginal, and the three results the paper
calls substantive -- Propositions `marginal` and `advbound` and Theorem
`bridge` -- are stated about a distinction the numbers cannot exhibit. E9
measured that EMBER-2018 alerts at ``alert_malware_fraction = 0.9378``, so its
alert population carries genuine benign mass and the distinction becomes
measurable.

Two choices here are made for comparability rather than convenience, and both
are recorded in the returned provenance dict:

* **The training set is E9's, exactly.** Same seed, same
  `EmberCorpus.stratified_sample`, same size as E1 used on BODMAS. A larger
  training set would confound "the corpus got harder" with "the model got more
  data", which is the question E9 was built to keep separate.
* **The test set is a stratified subsample.** EMBER-2018 ships 200k test rows;
  `predict_proba` converts to float64, so scoring all of them at full width
  costs ~3.8 GB on top of the training matrix. The subsample preserves the
  benign/malware ratio, so the alert population it induces is the same
  population, measured with a wider confidence interval. The realised AUC is
  reported next to E9's full-test AUC so the cost of the subsample is visible.

The split itself is EMBER's own released one (train 2018-01..10, test
2018-11..12), which is already temporal, so "fit on the past, alert on the
future" holds without re-cutting.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..config import ExperimentConfig
from .loader import EmberCorpus, load_ember2018
from ..pipeline import Substrate, build_substrate_from_split

__all__ = ["DEFAULT_CORPUS_DIR", "read_rows", "build_ember_substrate"]

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_DIR = Path(os.environ.get(
    "TXAI_EMBER_MATRIX",
    Path(os.environ.get("TXAI_DATA_DIR", Path(__file__).resolve().parents[3] / "data"))
    / "ember2018_matrix"))

#: Test rows to keep. 50k x 2381 float32 is 476 MB resident and about twice that
#: again transiently inside `predict_proba`; the full 200k would not fit beside
#: the training matrix on a 16 GB machine.
DEFAULT_TEST_SIZE = 50_000


def read_rows(corpus: EmberCorpus, idx: np.ndarray,
              block: int = 25_000) -> np.ndarray:
    """Gather rows `idx` from the memmap into one resident float32 array.

    Filling a pre-allocated array block by block keeps the peak at the size of
    the result. `corpus.X[idx]` in one shot would hold the memmap's read buffer
    and the result at the same time.
    """
    out = np.empty((len(idx), corpus.X.shape[1]), dtype=np.float32)
    for start in range(0, len(idx), block):
        rows = idx[start:start + block]
        out[start:start + len(rows)] = corpus.X[rows]
    return out


def build_ember_substrate(cfg: ExperimentConfig, detector_name: str, seed: int,
                          n_alerts: int,
                          corpus_dir: Path = DEFAULT_CORPUS_DIR,
                          train_size: Optional[int] = None,
                          test_size: int = DEFAULT_TEST_SIZE
                          ) -> Tuple[Substrate, dict]:
    """A Substrate on EMBER-2018, built by the same code path as BODMAS's.

    Args:
        cfg: the frozen run configuration.
        detector_name: passed to `detectors.build_detector`.
        seed: controls the subsampling, the alert draw and the model.
        n_alerts: how many raised alerts to keep for the explanation stages.
        corpus_dir: the directory `build_ember_matrix.py` wrote.
        train_size: rows to train on. None means "match E1", read from the
            stored BODMAS result rather than typed here.
        test_size: rows to score for the alert population.

    Returns:
        (substrate, provenance). The provenance dict records every deviation
        from the full corpus so a number from this substrate can never be
        reported as if it came from all of EMBER-2018.

    Raises:
        FileNotFoundError: when the corpus has not been vectorised.
        RuntimeError: when the vectorisation did not finish.
    """
    corpus = load_ember2018(Path(corpus_dir))
    train_idx = corpus.select(subset="train")
    test_idx = corpus.select(subset="test")

    if train_size is None:
        train_size = _e1_train_size()
    train_subsampled = train_size < len(train_idx)
    if train_subsampled:
        train_idx = corpus.stratified_sample(train_idx, n=train_size, seed=seed)

    test_subsampled = test_size < len(test_idx)
    n_test_full = len(test_idx)
    if test_subsampled:
        test_idx = corpus.stratified_sample(test_idx, n=test_size, seed=seed)

    logger.info("EMBER substrate: %d train rows, %d test rows (of %d)",
                len(train_idx), len(test_idx), n_test_full)

    X_train = read_rows(corpus, train_idx)
    y_train = np.asarray(corpus.y[train_idx]).astype(np.int64)
    X_test = read_rows(corpus, test_idx)
    y_test = np.asarray(corpus.y[test_idx]).astype(np.int64)

    sub = build_substrate_from_split(X_train, y_train, X_test, y_test,
                                     cfg, detector_name, seed, n_alerts)

    provenance = {
        "corpus": "EMBER-2018 feature version 2",
        # The directory *name*, not its path. A results file is published, and an
        # absolute path carries the operator's home directory into it; the name
        # is the part a reader can act on -- which matrix directory was read --
        # and the location is theirs to choose through `TXAI_EMBER_MATRIX`.
        "corpus_dir": Path(corpus_dir).name,
        "split": "EMBER's released temporal split (train 2018-01..10, "
                 "test 2018-11..12), used as shipped",
        "train_size": int(len(train_idx)),
        "train_subsampled": bool(train_subsampled),
        "train_subsample_reason": (
            "matched to E1's BODMAS training size so the corpus is the only "
            "variable between the two runs"),
        "test_size": int(len(test_idx)),
        "test_size_full": int(n_test_full),
        "test_subsampled": bool(test_subsampled),
        "test_subsample_reason": (
            "stratified, to bound resident memory; preserves the "
            "benign/malware ratio and therefore the alert population"),
        "train_malware_fraction": float((y_train == 1).mean()),
        "test_malware_fraction": float((y_test == 1).mean()),
        "auc": sub.auc,
        "alert_rate": sub.alert_rate,
        "n_alerts": int(len(sub.X_alerts)),
        "alert_malware_fraction": float((sub.y_alerts == 1).mean()),
        "alert_benign_count": int((sub.y_alerts == 0).sum()),
    }
    logger.info("EMBER substrate: AUC=%.6f, %d alerts, %.4f malware "
                "(%d benign)", sub.auc, provenance["n_alerts"],
                provenance["alert_malware_fraction"],
                provenance["alert_benign_count"])
    return sub, provenance


def _e1_train_size(default: int = 104_578) -> int:
    """E1's BODMAS training size, read from its artifact rather than hardcoded.

    A literal here would keep reading as true after E1 was re-run at a different
    size, and the comparability claim above would quietly become false.
    """
    import json

    path = (Path(__file__).resolve().parents[3] / "results" /
            "E1_substrate.json")
    if not path.exists():
        logger.warning("E1_substrate.json absent; falling back to %d", default)
        return default
    return int(json.loads(path.read_text()).get("train_size", default))
