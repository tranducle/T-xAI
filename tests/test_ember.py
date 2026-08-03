"""Tests for the EMBER-2018 vectorisation path.

The tests that matter here are the ones that would have caught a silent error:
a hashing convention that puts the vectors in a different feature space from the
corpus they are compared against, a resume that redoes or skips work, and a
loader that hands out a matrix with unwritten holes in it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from txai.config import N_FEATURES  # noqa: E402
from txai.ember.compat import ENTRY_NAME_CONVENTIONS, build_extractor  # noqa: E402
from txai.ember.loader import load_ember2018  # noqa: E402

import build_ember_matrix as ve  # noqa: E402

# entry_name_hashed occupies these columns: the sections block starts at 688 and
# is laid out as general(5) + sizes(50) + entropy(50) + vsize(50) + entry(50) +
# characteristics(50).
ENTRY_NAME_COLS = slice(688 + 5 + 150, 688 + 5 + 200)

#: Corpora are not redistributed here. Point ``TXAI_DATA_DIR`` at a directory
#: holding the EMBER-2018 JSONL release and ``bodmas.npz`` (see README, "Data");
#: everything that needs them skips when they are absent.
DATA_DIR = Path(os.environ.get("TXAI_DATA_DIR", ROOT / "data"))
JSONL_DIR = DATA_DIR / "ember2018"
VECTORISED = Path(os.environ.get("TXAI_EMBER_MATRIX", DATA_DIR / "ember2018_matrix"))

needs_jsonl = pytest.mark.skipif(
    not (JSONL_DIR / "train_features_0.jsonl").exists(),
    reason="EMBER-2018 JSONL not extracted")
needs_vectors = pytest.mark.skipif(
    not (VECTORISED / "manifest.json").exists(),
    reason="EMBER-2018 not vectorised")


@pytest.fixture(scope="module")
def sample_rows() -> list:
    with (JSONL_DIR / "train_features_0.jsonl").open() as fh:
        return [json.loads(next(fh)) for _ in range(200)]


# --------------------------------------------------------------------------
# The hashing convention
# --------------------------------------------------------------------------

def test_extractor_width_is_2381():
    assert build_extractor("chars").dim == N_FEATURES


def test_unknown_convention_rejected():
    with pytest.raises(ValueError, match="convention must be one of"):
        build_extractor("nonsense")


@needs_jsonl
def test_both_conventions_produce_full_width_vectors(sample_rows):
    for convention in ENTRY_NAME_CONVENTIONS:
        extractor = build_extractor(convention)
        vector = extractor.process_raw_features(sample_rows[0])
        assert vector.shape == (N_FEATURES,)
        assert vector.dtype == np.float32


@needs_jsonl
def test_chars_convention_matches_bodmas_entry_name_signature(sample_rows):
    """The convention is a fidelity question, and this is how it was settled.

    BODMAS was vectorised by the original EMBER extractor under the old
    scikit-learn, so its entry-name block carries that behaviour's fingerprint.
    Under `token` every row has exactly one non-zero, and a row with an empty
    entry name still hashes the empty string to one bin, so zero non-zeros is
    unreachable. BODMAS does have such rows. Only `chars` can produce them.
    """
    bodmas_path = DATA_DIR / "bodmas.npz"
    if not bodmas_path.exists():
        pytest.skip("BODMAS archive not present")

    bodmas = np.load(bodmas_path)["X"][:20000, ENTRY_NAME_COLS]
    bodmas_nz = (bodmas != 0).sum(axis=1)
    assert (bodmas_nz == 0).any(), "expected BODMAS rows with an empty entry name"
    assert np.median(bodmas_nz) == 4

    counts = {}
    for convention in ENTRY_NAME_CONVENTIONS:
        extractor = build_extractor(convention)
        block = np.vstack([extractor.process_raw_features(r)
                           for r in sample_rows])[:, ENTRY_NAME_COLS]
        counts[convention] = (block != 0).sum(axis=1)

    assert (counts["token"] == 1).all()
    assert np.median(counts["chars"]) == np.median(bodmas_nz)
    assert (counts["chars"] == 0).any()


# --------------------------------------------------------------------------
# Indexing and resume
# --------------------------------------------------------------------------

@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Path:
    """A 30-row source directory, so the vectoriser can run end to end."""
    if not (JSONL_DIR / "train_features_0.jsonl").exists():
        pytest.skip("EMBER-2018 JSONL not extracted")
    src = tmp_path / "src"
    src.mkdir()
    with (JSONL_DIR / "train_features_0.jsonl").open() as fh:
        lines = [next(fh) for _ in range(30)]
    (src / "train_features_0.jsonl").write_text("".join(lines[:20]))
    (src / "test_features.jsonl").write_text("".join(lines[20:]))
    return src


def test_month_index_round_trip():
    assert ve.month_index("1970-01") == 0
    assert ve.month_index("2018-11") == (2018 - 1970) * 12 + 10


@pytest.mark.parametrize("bad", ["", "2018", "not-a-date", None])
def test_month_index_flags_bad_values_rather_than_guessing(bad):
    assert ve.month_index(bad) == -1


def test_index_file_offsets_land_on_row_boundaries(tiny_corpus):
    path = tiny_corpus / "train_features_0.jsonl"
    n_rows, offsets = ve.index_file(path, chunk_rows=7)
    assert n_rows == 20
    assert len(offsets) == 3  # ceil(20 / 7)
    with path.open("rb") as fh:
        for offset in offsets:
            fh.seek(offset)
            json.loads(fh.readline())  # raises if the offset is mid-row


def test_end_to_end_and_resume(tiny_corpus, tmp_path):
    out = tmp_path / "out"
    assert ve.run(tiny_corpus, out, workers=2, chunk_rows=7,
                  limit_chunks=None) == 0

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_rows"] == 30
    n_chunks = len(manifest["chunks"])
    assert len(list((out / "chunks").glob("*.done"))) == n_chunks

    X = np.memmap(out / "X.dat", dtype="float32", mode="r",
                  shape=(30, N_FEATURES))
    assert not (np.asarray(X) == 0).all(axis=1).any()
    subset = np.memmap(out / "subset.dat", dtype="int8", mode="r", shape=(30,))
    assert int((np.asarray(subset) == 1).sum()) == 10

    # Re-running must be a no-op, and dropping one marker must redo exactly one
    # chunk. A resume that silently redoes everything is slow but safe; one that
    # silently skips unwritten rows is not, and both would pass a "exit 0" check.
    mtimes = {p.name: p.stat().st_mtime_ns
              for p in (out / "chunks").glob("*.done")}
    assert ve.run(tiny_corpus, out, workers=2, chunk_rows=7,
                  limit_chunks=None) == 0
    assert {p.name: p.stat().st_mtime_ns
            for p in (out / "chunks").glob("*.done")} == mtimes


def test_allocate_refuses_to_resume_onto_a_mismatched_file(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    ve.allocate(out, n_rows=10)
    with pytest.raises(ValueError, match="delete the output directory"):
        ve.allocate(out, n_rows=11)


# --------------------------------------------------------------------------
# The loader
# --------------------------------------------------------------------------

def test_loader_refuses_an_incomplete_run(tiny_corpus, tmp_path):
    """An unfinished matrix is zeros, which a model would read as real samples."""
    out = tmp_path / "out"
    ve.run(tiny_corpus, out, workers=2, chunk_rows=7, limit_chunks=None)
    next((out / "chunks").glob("*.done")).unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        load_ember2018(out)


@needs_vectors
def test_full_corpus_matches_the_published_ember2018_spec():
    corpus = load_ember2018(VECTORISED)
    assert corpus.X.shape == (1_000_000, N_FEATURES)
    assert int((corpus.subset == 0).sum()) == 800_000
    assert int((corpus.subset == 1).sum()) == 200_000
    assert int((corpus.y == 0).sum()) == 400_000
    assert int((corpus.y == 1).sum()) == 400_000
    assert int((corpus.y == -1).sum()) == 200_000
    # The released split is temporal: every test sample appeared after every
    # training sample.
    train_month = corpus.month[corpus.subset == 0].max()
    test_month = corpus.month[corpus.subset == 1].min()
    assert train_month < test_month


@needs_vectors
def test_select_drops_unlabeled_by_default():
    corpus = load_ember2018(VECTORISED, verify=False)
    assert len(corpus.select(subset="train")) == 600_000
    assert len(corpus.select(subset="train", labeled_only=False)) == 800_000


@needs_vectors
def test_stratified_sample_preserves_the_class_ratio():
    corpus = load_ember2018(VECTORISED, verify=False)
    idx = corpus.select(subset="train")
    sample = corpus.stratified_sample(idx, n=10_000, seed=42)
    assert abs(len(sample) - 10_000) <= 2
    ratio = (corpus.y[sample] == 1).mean()
    assert abs(ratio - (corpus.y[idx] == 1).mean()) < 0.01
    # Same seed, same rows: the subsample has to be reportable in the paper.
    assert np.array_equal(sample, corpus.stratified_sample(idx, 10_000, seed=42))


@needs_vectors
def test_temporal_cut_puts_the_future_in_the_test_half():
    corpus = load_ember2018(VECTORISED, verify=False)
    idx = corpus.select(subset="train", drop_unknown_month=True)
    idx = corpus.stratified_sample(idx, n=50_000, seed=0)
    train_idx, test_idx = corpus.temporal_cut(idx, test_fraction=0.2)
    assert corpus.month[train_idx].max() <= corpus.month[test_idx].min()
    assert len(train_idx) + len(test_idx) == len(idx)
