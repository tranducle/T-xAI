"""Read the vectorised EMBER-2018 corpus without loading it into memory.

`build_ember_matrix.py` writes the corpus as memmaps. Nothing here undoes that: the
feature matrix is returned as a `np.memmap`, so a full-corpus handle costs a few
kilobytes of address space rather than the ~9.5 GB the matrix occupies on disk.
On a 16 GB machine that distinction is the difference between running and
swapping.

The consequence to keep in mind is that **fancy indexing a memmap copies**.
`X[idx]` for 600k rows would materialise 5.7 GB in RAM. Selection is therefore
returned as index arrays, and `materialise_subset` writes a chosen subset to a new
memmap on disk rather than to memory.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..config import N_FEATURES
from ..data import verify_feature_layout

__all__ = ["EmberCorpus", "load_ember2018", "materialise_subset"]

logger = logging.getLogger(__name__)

SUBSET_TRAIN, SUBSET_TEST = 0, 1
LABEL_UNLABELED = -1


@dataclass(frozen=True)
class EmberCorpus:
    """On-disk EMBER-2018 corpus. `X` is a memmap and is never fully resident."""

    X: np.memmap          # float32 (N, 2381)
    y: np.ndarray         # int8   (N,)   -1 unlabeled / 0 benign / 1 malware
    month: np.ndarray     # int32  (N,)   months since 1970-01, -1 unknown
    subset: np.ndarray    # int8   (N,)   0 train / 1 test
    sha256: np.ndarray    # S64    (N,)
    avclass: np.ndarray   # S32    (N,)
    bad_rows: np.ndarray  # int64  indices left zero-filled by the vectoriser
    root: Path

    def __post_init__(self) -> None:
        n = len(self.X)
        for name in ("y", "month", "subset", "sha256", "avclass"):
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"{name} has {len(getattr(self, name))} rows but X has {n}")

    def select(self, subset: Optional[str] = None, labeled_only: bool = True,
               drop_unknown_month: bool = False) -> np.ndarray:
        """Return row indices matching the filters, without copying features.

        Args:
            subset: "train", "test", or None for both.
            labeled_only: drop the label -1 rows. EMBER-2018 ships 200k
                unlabeled training samples; including them in a supervised fit
                would treat "unknown" as a third class.
            drop_unknown_month: drop rows whose `appeared` field did not parse.
                Needed before a temporal split, where an unknown date would sort
                to one end and land entirely in one side of the cut.

        Returns:
            int64 indices into the corpus rows.
        """
        keep = np.ones(len(self.X), dtype=bool)
        if subset is not None:
            code = {"train": SUBSET_TRAIN, "test": SUBSET_TEST}[subset]
            keep &= self.subset == code
        if labeled_only:
            keep &= self.y != LABEL_UNLABELED
        if drop_unknown_month:
            keep &= self.month >= 0
        if len(self.bad_rows):
            keep[self.bad_rows] = False
        idx = np.flatnonzero(keep)
        logger.info("select(subset=%s, labeled_only=%s): %d of %d rows",
                    subset, labeled_only, len(idx), len(self.X))
        return idx

    def stratified_sample(self, idx: np.ndarray, n: int, seed: int = 42
                          ) -> np.ndarray:
        """Draw `n` rows from `idx`, preserving the benign/malware ratio.

        Returns `idx` unchanged when it already holds `n` rows or fewer, so a
        caller need not special-case a small corpus.
        """
        if n >= len(idx):
            return idx
        rng = np.random.default_rng(seed)
        labels = self.y[idx]
        chosen = []
        for value in np.unique(labels):
            pool = idx[labels == value]
            take = int(round(n * len(pool) / len(idx)))
            take = min(take, len(pool))
            chosen.append(rng.choice(pool, size=take, replace=False))
        out = np.sort(np.concatenate(chosen))
        logger.info("stratified sample: %d of %d rows (seed=%d)",
                    len(out), len(idx), seed)
        return out

    def temporal_cut(self, idx: np.ndarray, test_fraction: float = 0.2
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """Split `idx` by `appeared` month: train on the past, test on the future.

        EMBER-2018's own train/test split is already temporal (train 2018-01 to
        2018-10, test 2018-11 to 2018-12). This is for the case where a cut is
        wanted inside one subset, mirroring what `data.temporal_split` does for
        BODMAS so the two corpora are treated the same way.
        """
        if np.any(self.month[idx] < 0):
            raise ValueError(
                "idx contains rows with an unparsed month; pass "
                "drop_unknown_month=True to select() before splitting")
        order = idx[np.argsort(self.month[idx], kind="stable")]
        cut = int((1.0 - test_fraction) * len(order))
        logger.info("temporal cut at month index %d: train=%d test=%d",
                    self.month[order[cut]], cut, len(order) - cut)
        return order[:cut], order[cut:]


def load_ember2018(root: Path, verify: bool = True) -> EmberCorpus:
    """Open the vectorised corpus at `root` as memmaps.

    Args:
        root: the `--out` directory `build_ember_matrix.py` wrote.
        verify: re-check the EMBER-v2 column order on a sample of rows. Keep
            this on; it is the check that catches a wrong feature version or a
            partially written matrix before either reaches a result.

    Raises:
        FileNotFoundError: when the manifest or an array file is missing.
        RuntimeError: when the run did not finish, so the matrix has holes.
    """
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no manifest at {manifest_path}; run build_ember_matrix.py first")
    manifest = json.loads(manifest_path.read_text())
    n = manifest["n_rows"]

    n_done = len(list((root / "chunks").glob("*.done")))
    n_chunks = len(manifest["chunks"])
    if n_done != n_chunks:
        raise RuntimeError(
            f"vectorisation is incomplete: {n_done} of {n_chunks} chunks "
            "finished. The unfinished rows are still zeros, which would enter "
            "a model as a spurious dense cluster of identical samples. Resume "
            "with supervise_vectorize.sh before loading.")

    def _open(name: str, dtype: str, two_d: bool = False) -> np.memmap:
        shape = (n, N_FEATURES) if two_d else (n,)
        return np.memmap(root / name, dtype=dtype, mode="r", shape=shape)

    bad_path = root / "bad_rows.json"
    bad_rows = (np.asarray(json.loads(bad_path.read_text()), dtype=np.int64)
                if bad_path.exists() else np.empty(0, dtype=np.int64))
    if len(bad_rows):
        logger.warning("%d rows were unparsable and are excluded by select()",
                       len(bad_rows))

    corpus = EmberCorpus(
        X=_open("X.dat", "float32", two_d=True),
        y=np.asarray(_open("y.dat", "int8")),
        month=np.asarray(_open("month.dat", "int32")),
        subset=np.asarray(_open("subset.dat", "int8")),
        sha256=np.asarray(_open("sha256.dat", "S64")),
        avclass=np.asarray(_open("avclass.dat", "S32")),
        bad_rows=bad_rows,
        root=root,
    )

    if verify:
        # Sample across the whole matrix, not the first rows: a chunked writer
        # that failed late would leave a clean head and a zeroed tail.
        probe = np.linspace(0, n - 1, num=3000, dtype=np.int64)
        verify_feature_layout(np.asarray(corpus.X[probe]))

    logger.info("loaded EMBER-2018: X=%s on disk, %d train / %d test, "
                "%d benign / %d malware / %d unlabeled",
                corpus.X.shape, int((corpus.subset == SUBSET_TRAIN).sum()),
                int((corpus.subset == SUBSET_TEST).sum()),
                int((corpus.y == 0).sum()), int((corpus.y == 1).sum()),
                int((corpus.y == LABEL_UNLABELED).sum()))
    return corpus


def materialise_subset(corpus: EmberCorpus, idx: np.ndarray, out_dir: Path,
                       block: int = 20_000) -> Path:
    """Copy the rows `idx` into a new memmap on disk, block by block.

    The point is that the copy never becomes resident: at most `block` rows
    (about 190 MB at the default) are in memory at once. A plain `X[idx]` would
    allocate the whole subset in RAM instead.

    Returns:
        The directory holding `X.dat`, `y.dat`, `month.dat` and `meta.json` for
        the subset.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = len(idx)

    Xs = np.memmap(out_dir / "X.dat", dtype="float32", mode="w+",
                   shape=(m, N_FEATURES))
    for start in range(0, m, block):
        rows = idx[start:start + block]
        Xs[start:start + len(rows)] = corpus.X[rows]
        Xs.flush()
    del Xs

    np.asarray(corpus.y[idx]).tofile(out_dir / "y.dat")
    np.asarray(corpus.month[idx]).tofile(out_dir / "month.dat")
    (out_dir / "meta.json").write_text(json.dumps({
        "n_rows": m, "n_features": N_FEATURES, "source": str(corpus.root),
        "dtypes": {"X": "float32", "y": "int8", "month": "int32"},
        "n_benign": int((corpus.y[idx] == 0).sum()),
        "n_malware": int((corpus.y[idx] == 1).sum()),
    }, indent=1))
    logger.info("materialised %d rows -> %s (%.2f GB)", m, out_dir,
                m * N_FEATURES * 4 / 1e9)
    return out_dir
