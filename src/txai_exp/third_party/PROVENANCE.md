# Vendored third-party code

## `ember_features.py`

| Field | Value |
|---|---|
| Upstream | `https://raw.githubusercontent.com/elastic/ember/master/ember/features.py` |
| Retrieved | 2026-08-03 |
| SHA-256 | `db0d93bb1e1d4b28558e373c77194a82c3cae09d7a1aeb44eea5dd68e84f3ffd` |
| Lines | 556 |
| Licence | MIT (Elastic / Endgame, Anderson & Roth 2018) |
| Modified | **No.** Byte-identical to upstream, so the hash above is verifiable |

### Why vendored rather than `pip install ember`

The upstream package pins `lightgbm` and a `lief` range that does not resolve on
Python 3.14. Only one class is needed — `PEFeatureExtractor` — and only along the
path that does *not* touch `lief`, so the dependency closure is unnecessary.

### Why the LIEF version warning does not apply to this use

Constructing `PEFeatureExtractor(feature_version=2)` prints:

```
WARNING: EMBER feature version 2 were computed using lief version 0.9.0-
WARNING:   lief version 0.17.6-08dc3b7f found instead.
```

The warning concerns `raw_features(bytez)`, which parses a PE binary with LIEF and
whose output can shift between LIEF releases. This project never calls it. The
EMBER-2018 release ships the raw feature dictionaries already extracted, so the
only path used is `process_raw_features(raw_obj)` — pure `numpy` plus
`sklearn.feature_extraction.FeatureHasher`, with no LIEF call anywhere beneath it.
LIEF is imported at module scope and otherwise unused on this path.

Independent of this argument, `txai_exp.data.verify_feature_layout` re-checks the
resulting matrix by an arithmetic invariant (columns 0:256 and 256:512 each sum to
exactly 1 per row) and raises if the layout is not the one BODMAS uses, so a silent
version mismatch cannot reach the results.

### Verified on this machine (2026-08-03)

- `PEFeatureExtractor(feature_version=2).dim` → `2381`, matching `config.N_FEATURES`
  and the BODMAS column count.
