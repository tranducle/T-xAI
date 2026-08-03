#!/usr/bin/env python3
"""E9 -- does F7 survive a second, harder corpus?

F7 says BODMAS's near-perfect separability is a collection artefact: eight of the
nine EMBER feature groups reach AUC >= 0.986 *alone*. Every explanation result in
this paper is conditioned on that. E9 re-runs exactly the E1 measurement on
EMBER-2018, which its own authors state was assembled so that "the resultant
training and test sets would be harder for machine learning algorithms to
classify".

Two outcomes, both usable, and which one holds is not decided here:

  - Separability drops materially. F7 was corpus-specific, and the paper's
    threat-to-validity narrows to BODMAS.
  - Separability holds. F7 generalises: high separability is a property of static
    PE features, not of one corpus. That is the stronger claim for the threats
    section, but it is *not* "we validated on a hard substrate", and must not be
    written up as if it were.

Method, held identical to `pipeline.e1_detector` so the two are comparable:
detector `histgb` (HistGradientBoostingClassifier, max_iter=200, lr=0.1), per-group
models at max_iter=40, AUC on a held-out future. EMBER-2018's released split is
already temporal -- train up to 2018-10, test 2018-11 onward -- so it is used as
is rather than re-cut.

**Why the training set is subsampled, and why that is not a hidden caveat.**
The subsample is chosen to *match E1*, not to fill the machine. E1 trained on
104,578 BODMAS rows; E9 draws the same number from EMBER-2018, so the corpus is
the only variable that differs and a change in per-group AUC cannot be blamed on
having trained on more data. Training on all 600k labeled rows would have been
the worse experiment even with the memory to do it.

It also happens to be what this machine can afford, and the first attempt showed
why that is not a footnote. Run at 300k rows, stage 1 spent 73 minutes with its
worker threads at roughly 96% system time -- twelve minutes of kernel time each
against twenty-seven seconds of user time -- and was stopped without producing an
AUC. That is not slow arithmetic, it is a 16 GB machine already 13 GB into swap
paying for a working set it cannot hold. Measured marginal cost is ~17.5 KB per
training row, so 300k rows need ~6.2 GB while E1's 104,578 need ~2.8 GB.

Stage 3 measures a learning curve at 25k / 50k / 100k rows so the size is a
quantified caveat rather than an asserted one: it shows what AUC does as training
data grows toward the size actually used, and whether the per-group result is
anywhere near a regime where more data would change it.

Everything is written to disk as it completes. A killed run resumes from the
results file rather than redoing finished stages, which matters because the full
fit is the expensive part.

Usage:
    python3 run_ember_transfer.py [--train-size N] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "src"))

from txai.config import EMBER_GROUPS  # noqa: E402
from txai.ember.loader import load_ember2018  # noqa: E402

logger = logging.getLogger("e9")

#: The vectorised EMBER-2018 matrix written by ``build_ember_matrix.py``.
#: Not redistributed here (see README, "Data"); override with
#: ``TXAI_EMBER_MATRIX``.
CORPUS_DIR = Path(os.environ.get(
    "TXAI_EMBER_MATRIX",
    Path(os.environ.get("TXAI_DATA_DIR", HERE.parents[0] / "data"))
    / "ember2018_matrix"))
RESULTS_DIR = HERE.parents[0] / "results"
RESULTS_PATH = RESULTS_DIR / "E9_ember2018.json"
E1_PATH = RESULTS_DIR / "E1_substrate.json"

# The threshold the manuscript states (paper.tex, sec:numlimits): "eight of the
# nine EMBER feature groups reach an AUC of at least 0.986 on their own". It is
# 0.986 and not 0.987 for a reason worth keeping visible -- byte_entropy lands at
# 0.986989 on BODMAS, so at 0.987 the count is seven, not eight. E9 must apply
# the same threshold to both corpora or the replication compares two different
# questions.
SEPARABILITY_TAU = 0.986


def e1_train_size(default: int = 104_578) -> int:
    """The training-set size E1 used on BODMAS, which E9 matches deliberately.

    Taking all 600k labeled EMBER rows would train the replication on six times
    the data E1 had, and then a difference in per-group AUC between the two
    corpora could be either the corpus or the training size. Holding the size
    fixed leaves the corpus as the only thing that changed, which is the whole
    question F7 asks. It is also what makes the run affordable: at this size the
    training matrix is 1.0 GB and the measured peak ~2.8 GB, against ~6.2 GB at
    300k rows on a machine that has 16 GB and is sharing them.

    Stage 3's learning curve then shows what the size costs in absolute AUC.
    """
    if not E1_PATH.exists():
        return default
    return int(json.loads(E1_PATH.read_text()).get("train_size", default))


def bodmas_reference() -> tuple:
    """How many BODMAS groups clear the threshold, read from E1 rather than typed.

    A hardcoded "8 of 9" would keep reading as true after E1 was re-run with a
    different seed or detector. Recomputing it from the stored artifact means the
    comparison cannot silently go stale.
    """
    if not E1_PATH.exists():
        return "unknown (E1_substrate.json absent)", "?"
    groups = json.loads(E1_PATH.read_text()).get(
        "auc_by_feature_group_alone", {})
    if not groups:
        return "unknown (no per-group AUCs in E1)", "?"
    return sum(1 for v in groups.values() if v >= SEPARABILITY_TAU), len(groups)


def peak_rss_gb() -> float:
    """Peak resident set size in GB. Linux reports kB, macOS reports bytes."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e9 if sys.platform == "darwin" else raw / 1e6


def read_columns(corpus, idx: np.ndarray, lo: int, hi: int,
                 block: int = 50_000) -> np.ndarray:
    """Gather rows `idx`, columns `[lo, hi)`, without materialising all columns.

    `corpus.X[idx][:, lo:hi]` would build the full-width copy first: 5.7 GB for
    600k rows, on top of whatever is already held. Filling a pre-allocated array
    block by block keeps the peak at the size of the slice actually wanted.

    float32 is deliberate and was measured rather than assumed. The suspicion was
    that `HistGradientBoostingClassifier` validates X as float64 and would hold a
    float64 duplicate alongside our float32 original; if so, allocating float64
    here would have been cheaper. Measured on this machine at 40k x 2381, in one
    process per dtype: float32 input peaks at 1.66 GB and fitting grows the peak
    by 0.32 GB, while float64 input peaks at 2.26 GB and grows it by 0.54 GB. A
    full float64 duplicate would have cost 0.76 GB, so this sklearn bins straight
    from float32 and never makes one. float32 is the cheaper input, and the
    hypothesis that sent this the other way was wrong.
    """
    out = np.empty((len(idx), hi - lo), dtype=np.float32)
    for start in range(0, len(idx), block):
        rows = idx[start:start + block]
        out[start:start + len(rows)] = corpus.X[rows, lo:hi]
    return out


def load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {}


def save_results(results: dict) -> None:
    """Write after every stage, so a kill costs one stage rather than the run."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=1))
    tmp.replace(RESULTS_PATH)  # atomic: a crash mid-write cannot truncate results


def batched_scores(clf, corpus, idx: np.ndarray, lo: int, hi: int,
                   block: int = 25_000) -> np.ndarray:
    """Score `idx` in blocks, so the test features are never all resident.

    `predict_proba` converts its input to float64, so handing it the whole test
    set would cost the float32 copy plus twice that again — 5.7 GB for 200k rows
    at full width, on top of the training arrays.
    """
    out = np.empty(len(idx), dtype=np.float64)
    for start in range(0, len(idx), block):
        rows = idx[start:start + block]
        out[start:start + len(rows)] = clf.predict_proba(
            np.asarray(corpus.X[rows, lo:hi]))[:, 1]
    return out


def fit_model(X_train: np.ndarray, y_train: np.ndarray, seed: int,
              max_iter: int) -> tuple:
    from sklearn.ensemble import HistGradientBoostingClassifier

    clf = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.1, random_state=seed)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    return clf, time.perf_counter() - t0


def run(train_size: Optional[int], seed: int) -> int:
    corpus = load_ember2018(CORPUS_DIR)
    train_idx = corpus.select(subset="train")
    test_idx = corpus.select(subset="test")

    subsampled = train_size is not None and train_size < len(train_idx)
    if subsampled:
        train_idx = corpus.stratified_sample(train_idx, n=train_size, seed=seed)
        logger.warning("training on a stratified subsample of %d rows; this is "
                       "a declared deviation and must be reported alongside "
                       "every number below", len(train_idx))

    results = load_results()
    results.setdefault("corpus", "EMBER-2018 feature version 2")
    results.setdefault("detector", "histgb")
    results.setdefault("seed", seed)
    results["train_size"] = int(len(train_idx))
    results["test_size"] = int(len(test_idx))
    results["train_subsampled"] = bool(subsampled)
    results["train_malware_fraction"] = float((corpus.y[train_idx] == 1).mean())
    results["test_malware_fraction"] = float((corpus.y[test_idx] == 1).mean())
    save_results(results)

    y_train = np.asarray(corpus.y[train_idx])
    y_test = np.asarray(corpus.y[test_idx])

    from sklearn.metrics import roc_auc_score
    n_cols = corpus.X.shape[1]

    # --- Stage 1: the full detector -------------------------------------
    if "auc" not in results:
        logger.info("stage 1: full detector on %d x %d", len(train_idx), n_cols)
        X_train = read_columns(corpus, train_idx, 0, n_cols)
        logger.info("  train %.2f GB resident, peak RSS %.2f GB",
                    X_train.nbytes / 1e9, peak_rss_gb())

        clf, fit_seconds = fit_model(X_train, y_train, seed, max_iter=200)
        # Free the training matrix before scoring: the model no longer needs it,
        # and holding both is what pushes the peak past what this machine has.
        del X_train
        scores = batched_scores(clf, corpus, test_idx, 0, n_cols)

        results["auc"] = float(roc_auc_score(y_test, scores))
        results["fit_seconds"] = fit_seconds
        results["alert_rate"] = float((scores >= 0.5).mean())
        results["alert_malware_fraction"] = float(
            (y_test[scores >= 0.5] == 1).mean())
        results["peak_rss_gb_stage1"] = round(peak_rss_gb(), 2)
        save_results(results)
        logger.info("  AUC=%.6f alert_rate=%.4f fit=%.1fs peak RSS %.2f GB",
                    results["auc"], results["alert_rate"], fit_seconds,
                    peak_rss_gb())
        del clf, scores
    else:
        logger.info("stage 1 already done: AUC=%.6f", results["auc"])

    # --- Stage 2: each feature group alone ------------------------------
    by_group = results.setdefault("auc_by_feature_group_alone", {})
    for name, (lo, hi) in EMBER_GROUPS.items():
        if name in by_group:
            logger.info("group %-18s already done: AUC=%.4f", name,
                        by_group[name])
            continue
        t0 = time.perf_counter()
        Xg_train = read_columns(corpus, train_idx, lo, hi)
        clf, _ = fit_model(Xg_train, y_train, seed, max_iter=40)
        del Xg_train
        scores = batched_scores(clf, corpus, test_idx, lo, hi)
        by_group[name] = float(roc_auc_score(y_test, scores))
        save_results(results)
        logger.info("group %-18s (%4d cols) AUC=%.6f  %.1fs  peak RSS %.2f GB",
                    name, hi - lo, by_group[name], time.perf_counter() - t0,
                    peak_rss_gb())
        del clf, scores

    # --- Stage 3: learning curve ----------------------------------------
    # Turns "we subsampled because of RAM" from an unquantified caveat into a
    # measured one. If AUC is already flat below the size actually used, the
    # ceiling on training rows is not what limits the finding.
    curve = results.setdefault("learning_curve", {})
    for n in (25_000, 50_000, 100_000, 200_000):
        if n >= len(train_idx) or str(n) in curve:
            continue
        sub_idx = corpus.stratified_sample(train_idx, n=n, seed=seed)
        Xs = read_columns(corpus, sub_idx, 0, n_cols)
        clf, _ = fit_model(Xs, np.asarray(corpus.y[sub_idx]), seed, max_iter=200)
        del Xs
        scores = batched_scores(clf, corpus, test_idx, 0, n_cols)
        curve[str(n)] = float(roc_auc_score(y_test, scores))
        save_results(results)
        logger.info("learning curve n=%7d AUC=%.6f", n, curve[str(n)])
        del clf, scores
    curve[str(len(train_idx))] = results["auc"]

    results["peak_rss_gb"] = round(peak_rss_gb(), 2)
    save_results(results)

    strong = [k for k, v in by_group.items() if v >= SEPARABILITY_TAU]
    bodmas_strong, bodmas_total = bodmas_reference()
    results["separability_tau"] = SEPARABILITY_TAU
    results["groups_at_or_above_tau"] = len(strong)
    results["bodmas_groups_at_or_above_tau"] = bodmas_strong
    save_results(results)
    logger.info("E9 complete: full AUC=%.6f, %d of %d groups reach AUC>=%.3f "
                "alone (BODMAS: %s of %s at the same threshold)",
                results["auc"], len(strong), len(by_group), SEPARABILITY_TAU,
                bodmas_strong, bodmas_total)
    logger.info("weakest groups: %s",
                sorted(by_group.items(), key=lambda kv: kv[1])[:3])
    logger.info("wrote %s", RESULTS_PATH)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=e1_train_size(),
                        help="stratified subsample size. The default matches "
                             "E1's BODMAS training size, so corpus is the only "
                             "variable that differs between the two runs — see "
                             "e1_train_size(). Pass 0 for all 600k labeled rows "
                             "on a machine with more memory than this one.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log", type=Path,
                        default=HERE / "results" / "e9.log")
    args = parser.parse_args(argv)

    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, force=True,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(args.log)])
    return run(args.train_size or None, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
