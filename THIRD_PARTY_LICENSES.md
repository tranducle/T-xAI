# Third-party code and data

## Vendored code

### `src/txai_exp/third_party/ember_features.py`

| Field | Value |
|---|---|
| Upstream | <https://github.com/elastic/ember> — `ember/features.py` |
| Licence | MIT |
| Modification | **None.** Byte-identical to upstream |
| SHA-256 | `db0d93bb1e1d4b28558e373c77194a82c3cae09d7a1aeb44eea5dd68e84f3ffd` |

The upstream copyright notice and full licence text are in the `LICENSE` file of
the upstream repository linked above; they are not reproduced here in order to
avoid transcribing a notice this repository has not verified character by
character. `src/txai_exp/third_party/PROVENANCE.md` records why the file is vendored
rather than installed, and how to re-verify the hash:

```bash
shasum -a 256 src/txai_exp/third_party/ember_features.py
```

## Datasets — referenced, not redistributed

Neither corpus is included in this repository, in any form. Both are obtained
from their publishers under the publishers' own terms; see the README, "Data".

| Corpus | Source | Used for |
|---|---|---|
| BODMAS | <https://whyisyoung.github.io/BODMAS/> | the primary instantiation (E1–E8) |
| EMBER-2018 | <https://github.com/elastic/ember> | the second-corpus replication (E9) |

Nothing in this repository writes into a dataset directory: every load path is
read-only, and the location is supplied by the reader through `TXAI_DATA_DIR`.
