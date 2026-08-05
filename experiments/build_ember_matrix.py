#!/usr/bin/env python3
"""Vectorise the EMBER-2018 v2 release into on-disk memmaps.

Everything this script produces is written straight to disk. No stage holds the
full matrix in memory: the largest live allocation is one chunk, 5000 rows by
2381 float32 columns, about 47 MB per worker.

The run is resumable at chunk granularity. Each finished chunk drops a marker in
`chunks/`, and a restart skips every chunk that already has one, so a crash or a
kill costs at most the work in flight. That is what makes the supervisor's
retry-and-resume loop safe.

Two passes:

  1. Index. Read every JSONL file once without parsing, count its lines, and
     record a byte offset every `--chunk-rows` lines. Written to `manifest.json`
     and reused on restart. Costs one sequential read of ~10 GB.
  2. Vectorise. Workers take (file, byte offset, row range) tasks and write into
     disjoint slices of shared memmaps. Disjointness is what makes the
     concurrent writes safe without a lock.

Outputs, in `--out`:

    X.dat        float32 (N, 2381)  EMBER feature-version-2 vectors
    y.dat        int8    (N,)       -1 unlabeled, 0 benign, 1 malware
    month.dat    int32   (N,)       months since 1970-01 from `appeared`, -1 unknown
    subset.dat   int8    (N,)       0 train, 1 test
    sha256.dat   S64     (N,)       sample hash, for provenance and overlap checks
    avclass.dat  S32     (N,)       AVClass family label, empty where absent
    manifest.json                   shapes, dtypes, row counts, chunk index

Usage:
    python3 build_ember_matrix.py --src <jsonl dir> --out <output dir> [--workers 7]
    python3 build_ember_matrix.py --src ... --out ... --limit-chunks 2   # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from txai_exp.config import N_FEATURES  # noqa: E402

logger = logging.getLogger("build_ember_matrix")

TRAIN_GLOB = "train_features_*.jsonl"
TEST_NAME = "test_features.jsonl"

SHA_WIDTH = 64
AVCLASS_WIDTH = 32

# Written once by the parent before any worker starts, read by every worker.
ARRAY_SPEC = {
    "X": ("X.dat", "float32", ("N", N_FEATURES)),
    "y": ("y.dat", "int8", ("N",)),
    "month": ("month.dat", "int32", ("N",)),
    "subset": ("subset.dat", "int8", ("N",)),
    "sha256": ("sha256.dat", f"S{SHA_WIDTH}", ("N",)),
    "avclass": ("avclass.dat", f"S{AVCLASS_WIDTH}", ("N",)),
}


@dataclass(frozen=True)
class Chunk:
    """One unit of resumable work: a contiguous run of rows in one file."""

    chunk_id: int
    path: str
    byte_offset: int
    n_rows: int
    global_start: int
    subset: int

    @property
    def marker_name(self) -> str:
        return f"{self.chunk_id:05d}.done"


# --------------------------------------------------------------------------
# Pass 1: index
# --------------------------------------------------------------------------

def index_file(path: Path, chunk_rows: int) -> Tuple[int, List[int]]:
    """Count lines and record a byte offset every `chunk_rows` lines.

    Reads bytes without decoding or parsing JSON, so this is I/O bound rather
    than CPU bound.

    Returns:
        (total_rows, offsets) where offsets[i] is the byte position at which row
        `i * chunk_rows` begins.
    """
    offsets: List[int] = []
    n_rows = 0
    with path.open("rb") as fh:
        while True:
            if n_rows % chunk_rows == 0:
                offsets.append(fh.tell())
            line = fh.readline()
            if not line:
                break
            # A trailing newline at EOF yields b"" above, so any non-empty read
            # is a real row; blank lines would be malformed input.
            if line.strip():
                n_rows += 1
            else:
                logger.warning("%s: blank line at byte %d ignored",
                               path.name, fh.tell())
    # The loop appends one offset past the final row when the count lands on a
    # chunk boundary; trim so len(offsets) == ceil(n_rows / chunk_rows).
    n_needed = (n_rows + chunk_rows - 1) // chunk_rows
    return n_rows, offsets[:n_needed]


def build_manifest(src: Path, out: Path, chunk_rows: int) -> dict:
    """Index every source file, or reuse the cached index if one exists."""
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("chunk_rows") == chunk_rows:
            logger.info("reusing cached index: %d rows, %d chunks",
                        manifest["n_rows"], len(manifest["chunks"]))
            return manifest
        logger.warning("cached index used chunk_rows=%s, rebuilding for %d",
                       manifest.get("chunk_rows"), chunk_rows)

    files: List[Tuple[Path, int]] = [
        (p, 0) for p in sorted(src.glob(TRAIN_GLOB))
    ]
    test_path = src / TEST_NAME
    if test_path.exists():
        files.append((test_path, 1))
    if not files:
        raise FileNotFoundError(f"no EMBER JSONL files under {src}")

    chunks: List[Chunk] = []
    cursor = 0
    for path, subset in files:
        t0 = time.time()
        n_rows, offsets = index_file(path, chunk_rows)
        for i, byte_offset in enumerate(offsets):
            rows_here = min(chunk_rows, n_rows - i * chunk_rows)
            chunks.append(Chunk(chunk_id=len(chunks), path=str(path),
                                byte_offset=byte_offset, n_rows=rows_here,
                                global_start=cursor, subset=subset))
            cursor += rows_here
        logger.info("indexed %-24s %8d rows in %5.1fs (%s)",
                    path.name, n_rows, time.time() - t0,
                    "test" if subset else "train")

    manifest = {
        "src": str(src),
        "chunk_rows": chunk_rows,
        "n_rows": cursor,
        "n_features": N_FEATURES,
        "arrays": {k: {"file": f, "dtype": d} for k, (f, d, _) in ARRAY_SPEC.items()},
        "chunks": [asdict(c) for c in chunks],
    }
    manifest_path.write_text(json.dumps(manifest, indent=1))
    logger.info("wrote index: %d rows, %d chunks -> %s",
                cursor, len(chunks), manifest_path)
    return manifest


# --------------------------------------------------------------------------
# Pass 2: vectorise
# --------------------------------------------------------------------------

_EXTRACTOR = None
_CONVENTION = None


def _get_extractor(convention: str):
    """Build the extractor once per worker process.

    The extractor is built through `txai_exp.ember.compat`, which restores the
    hashing semantics the published EMBER vectors were computed under; see that
    module for why the upstream file is not simply corrected. The LIEF version
    warning printed by `PEFeatureExtractor.__init__` does not apply on this code
    path — see `txai_exp/third_party/PROVENANCE.md` — and is suppressed there.
    """
    global _EXTRACTOR, _CONVENTION
    if _EXTRACTOR is None or _CONVENTION != convention:
        from txai_exp.ember.compat import build_extractor
        _EXTRACTOR = build_extractor(convention)
        _CONVENTION = convention
    return _EXTRACTOR


def month_index(appeared: str) -> int:
    """Convert EMBER's `appeared` field ("YYYY-MM") to months since 1970-01.

    Returns -1 when the field is missing or malformed, so a bad value is
    droppable downstream rather than silently sorting to one end of a split.
    """
    try:
        year, month = appeared.split("-")
        return (int(year) - 1970) * 12 + (int(month) - 1)
    except (AttributeError, ValueError):
        return -1


def _read_rows(path: str, byte_offset: int, n_rows: int) -> Iterator[str]:
    """Yield exactly `n_rows` raw lines starting at `byte_offset`."""
    with open(path, "rb") as fh:
        fh.seek(byte_offset)
        emitted = 0
        while emitted < n_rows:
            line = fh.readline()
            if not line:
                raise EOFError(
                    f"{path}: hit EOF after {emitted} of {n_rows} rows from "
                    f"byte {byte_offset}; the cached index is stale")
            if line.strip():
                emitted += 1
                yield line.decode("utf-8")


def process_chunk(chunk_dict: dict, out_dir: str, n_rows_total: int,
                  convention: str = "chars") -> dict:
    """Vectorise one chunk and write it into its slice of the memmaps.

    Runs in a worker process. Writes are confined to
    `[global_start, global_start + n_rows)`, which no other chunk touches, so the
    concurrent writers need no lock.

    Returns:
        A summary dict; `bad_rows` lists global row indices that failed to parse
        and were left as zeros.
    """
    chunk = Chunk(**chunk_dict)
    out = Path(out_dir)
    marker = out / "chunks" / chunk.marker_name
    if marker.exists():
        return {"chunk_id": chunk.chunk_id, "skipped": True, "bad_rows": []}

    extractor = _get_extractor(convention)
    t0 = time.time()

    X = np.zeros((chunk.n_rows, N_FEATURES), dtype=np.float32)
    y = np.full(chunk.n_rows, -1, dtype=np.int8)
    month = np.full(chunk.n_rows, -1, dtype=np.int32)
    sha = np.zeros(chunk.n_rows, dtype=f"S{SHA_WIDTH}")
    avclass = np.zeros(chunk.n_rows, dtype=f"S{AVCLASS_WIDTH}")
    bad_rows: List[int] = []

    for i, line in enumerate(_read_rows(chunk.path, chunk.byte_offset,
                                        chunk.n_rows)):
        try:
            raw = json.loads(line)
            X[i] = extractor.process_raw_features(raw)
            y[i] = int(raw.get("label", -1))
            month[i] = month_index(raw.get("appeared", ""))
            sha[i] = str(raw.get("sha256", ""))[:SHA_WIDTH].encode()
            avclass[i] = str(raw.get("avclass", ""))[:AVCLASS_WIDTH].encode()
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            # Leave the row as zeros and record it. Shrinking the chunk instead
            # would shift every later row's global index.
            bad_rows.append(chunk.global_start + i)
            logger.warning("chunk %d row %d unparsable: %s",
                           chunk.chunk_id, chunk.global_start + i, exc)

    lo, hi = chunk.global_start, chunk.global_start + chunk.n_rows
    for name, array in (("X", X), ("y", y), ("month", month),
                        ("sha256", sha), ("avclass", avclass)):
        fname, dtype, shape = ARRAY_SPEC[name]
        full_shape = ((n_rows_total, N_FEATURES) if len(shape) == 2
                      else (n_rows_total,))
        mm = np.memmap(out / fname, dtype=dtype, mode="r+", shape=full_shape)
        mm[lo:hi] = array
        mm.flush()
        del mm

    fname, dtype, _ = ARRAY_SPEC["subset"]
    mm = np.memmap(out / fname, dtype=dtype, mode="r+", shape=(n_rows_total,))
    mm[lo:hi] = np.int8(chunk.subset)
    mm.flush()
    del mm

    marker.write_text(json.dumps({
        "chunk_id": chunk.chunk_id, "n_rows": chunk.n_rows,
        "global_start": chunk.global_start, "bad_rows": bad_rows,
        "seconds": round(time.time() - t0, 2),
    }))
    return {"chunk_id": chunk.chunk_id, "skipped": False, "bad_rows": bad_rows,
            "seconds": time.time() - t0}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def allocate(out: Path, n_rows: int) -> None:
    """Create each output file at full size if it does not already exist.

    `np.memmap(mode="w+")` would truncate an existing file and destroy the work
    a resumed run is trying to keep, so existing files of the right size are
    left alone and a wrong-sized one is a hard error rather than a silent
    reallocation.
    """
    for name, (fname, dtype, shape) in ARRAY_SPEC.items():
        path = out / fname
        full_shape = (n_rows, N_FEATURES) if len(shape) == 2 else (n_rows,)
        expected = int(np.dtype(dtype).itemsize * np.prod(full_shape))
        if path.exists():
            actual = path.stat().st_size
            if actual != expected:
                raise ValueError(
                    f"{path} is {actual} bytes but this run needs {expected}; "
                    "delete the output directory to start over rather than "
                    "resuming onto a mismatched file")
            continue
        mm = np.memmap(path, dtype=dtype, mode="w+", shape=full_shape)
        mm.flush()
        del mm
        logger.info("allocated %-12s %s %s = %.2f GB",
                    fname, dtype, full_shape, expected / 1e9)


def run(src: Path, out: Path, workers: int, chunk_rows: int,
        limit_chunks: Optional[int], convention: str = "chars") -> int:
    out.mkdir(parents=True, exist_ok=True)
    (out / "chunks").mkdir(exist_ok=True)

    manifest = build_manifest(src, out, chunk_rows)
    n_rows = manifest["n_rows"]

    free = os.statvfs(out).f_bavail * os.statvfs(out).f_frsize
    need = sum(int(np.dtype(d).itemsize
                   * (n_rows * N_FEATURES if len(s) == 2 else n_rows))
               for _, d, s in ARRAY_SPEC.values())
    logger.info("disk: need %.1f GB, free %.1f GB", need / 1e9, free / 1e9)
    if need > free:
        logger.error("not enough free disk: need %.1f GB, have %.1f GB",
                     need / 1e9, free / 1e9)
        return 2

    allocate(out, n_rows)

    chunks: Sequence[dict] = manifest["chunks"]
    if limit_chunks:
        chunks = chunks[:limit_chunks]
        logger.warning("SMOKE TEST: only the first %d of %d chunks will run; "
                       "the output is incomplete and must not be used as "
                       "evidence", limit_chunks, len(manifest["chunks"]))

    pending = [c for c in chunks
               if not (out / "chunks" / f"{c['chunk_id']:05d}.done").exists()]
    done_already = len(chunks) - len(pending)
    logger.info("%d chunks total, %d already done, %d to run, %d workers",
                len(chunks), done_already, len(pending), workers)
    if not pending:
        logger.info("nothing to do")
        return 0

    t0 = time.time()
    completed, failed, all_bad = done_already, 0, []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_chunk, c, str(out), n_rows, convention): c
                   for c in pending}
        for fut in as_completed(futures):
            chunk = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001 - report, keep the pool alive
                failed += 1
                logger.error("chunk %d FAILED: %s: %s", chunk["chunk_id"],
                             type(exc).__name__, exc)
                continue
            completed += 1
            all_bad.extend(result["bad_rows"])
            if completed % 10 == 0 or completed == len(chunks):
                elapsed = time.time() - t0
                rate = (completed - done_already) / max(elapsed, 1e-9)
                remaining = (len(chunks) - completed) / max(rate, 1e-9)
                logger.info("progress %d/%d chunks (%.1f%%) elapsed %.1f min, "
                            "eta %.1f min", completed, len(chunks),
                            100 * completed / len(chunks), elapsed / 60,
                            remaining / 60)

    logger.info("finished: %d/%d chunks, %d failed, %d unparsable rows, "
                "%.1f min", completed, len(chunks), failed, len(all_bad),
                (time.time() - t0) / 60)
    if all_bad:
        (out / "bad_rows.json").write_text(json.dumps(sorted(all_bad)))
        logger.warning("wrote %d unparsable row indices to bad_rows.json; "
                       "they are zero-filled and must be dropped downstream",
                       len(all_bad))
    return 1 if failed else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True,
                        help="directory holding the EMBER JSONL files")
    parser.add_argument("--out", type=Path, required=True,
                        help="directory for the memmaps and the index")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--chunk-rows", type=int, default=5000)
    parser.add_argument("--limit-chunks", type=int, default=None,
                        help="smoke test: run only the first N chunks")
    parser.add_argument("--convention", default="chars",
                        choices=("chars", "token"),
                        help="how to hash the entry section name; see "
                             "txai_exp/ember/compat.py")
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args(argv)

    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log))
    logging.basicConfig(
        level=logging.INFO, handlers=handlers, force=True,
        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    return run(args.src, args.out, args.workers, args.chunk_rows,
               args.limit_chunks, args.convention)


if __name__ == "__main__":
    raise SystemExit(main())
