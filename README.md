# T-XAI — A Threat-Model-Aware Contract for XAI Explanation Release in Cybersecurity

> **Anonymity.** This repository is deliberately author-free: no names, affiliations,
> acknowledgements, funding statements, institutional paths, or contact addresses appear
> anywhere in the code, documentation, or recorded results. Please keep it that way in any
> issue or pull request until review concludes.

---

## What the paper asks

Explainable AI is often proposed as a route to transparent security analytics. But a security
explanation is an *operational artifact*: an adversary may observe it, manipulate it, or exploit
it, and the literature states no contract a deployment can be checked against. Holding the
detector fixed, this work asks what an explanation must satisfy before it can serve as auditable
evidence for a permitted security action under an explicit adversary model.

T-XAI is defined as an explanation contract binding an explanation to attacker capability,
stakeholder role, evidence obligation, disclosure exposure, and response actionability. On it the
paper builds an explanation-risk functional with four admissibility conditions, four threat
classes with an audit protocol, and a Stackelberg release game whose role-indexed release and
response policies are committed to before the adversary acts.

**This repository is the measurement layer only.** It computes the faithfulness ($F$), robustness
($B$) and disclosure ($D$) scores, the Equation (7) calibration condition, both sides of the
Theorem 1 bridge, and the release game's equilibrium. It does **not** implement the governance
layer — provenance, the claim–evidence graph, a release lattice with real consequences, or
actionability — because the public corpora used here carry no telemetry provenance, no analyst
roles, and no response playbook. That boundary is stated in `src/txai_exp/__init__.py` and is not
negotiable downstream: no write-up should describe this code as instantiating more than it does.

---

## Layout

```text
src/txai_exp/              measurement package
  config.py                experiment configuration; EMBER-v2 feature-group offsets
  data.py                  BODMAS loading, layout verification, temporal splitting
  detectors.py             detector families and the score function
  explainers.py            TreeSHAP, LIME, and the two controls (constant, random)
  metrics.py               F, B, D, sufficiency Phi, admissibility, deletion curves
  perturbations.py         adversary perturbations and their invariants
  release.py               response kernels, Adv^TV, the Theorem 1 bridge check
  game.py                  the Stackelberg release game
  pipeline.py              stage orchestration
  supplementary.py         the four run-3 measurements over a built substrate
  controls.py              positive controls: inject a known defect, show it is caught
  alt_metrics.py           alternative instantiations of D and F, for the sweep
  actionability.py         a minimal computable A_Gamma (proxy; see docs/ROUND3_EMBER.md)
  calibration_sweep.py     relaxing Phi = 1 to Phi >= theta; the calibration frontier
  ember/                   EMBER-2018 support (compatibility, loader, substrate)
  third_party/             vendored upstream code, byte-identical, with SHA-256

experiments/               drivers; each writes results/<name>.json and nothing else
  run_scoring_layer.py            main programme: E1, E2, E5, E6, E7 + ablations
  run_calibration_falsification.py  E3, the strict reading of Eq. (7)
  run_self_stability.py           E4b, explainer nondeterminism
  run_kernel_constant.py          E5b, the Dobrushin constant
  run_release_game.py             E8, the Stackelberg equilibrium sweep
  build_ember_matrix.py           vectorise the EMBER-2018 JSONL release
  run_ember_transfer.py           E9, the second-corpus replication
  run_numerical_instantiation.py  the whole numerical section, on either corpus
  run_ember_ablations.py          seed and detector ablations on EMBER-2018
  run_release_game_ember.py       E8 on EMBER-2018, with the marginal counterfactual
  supervise_ember_transfer.sh     retry/resume supervisor for E9
  watch_ember_transfer.sh         read-only progress monitor for E9

results/                   recorded outputs — the paper's evidence
docs/
  PROTOCOL.md              the experiment programme, costed before it was run
  EVIDENCE_GATE.md         the pre-registration gate and its claim-narrowing verdict
  RESULTS.md               run 2 (BODMAS): findings F1-F8 and the claim-to-evidence matrix
  ROUND3_EMBER.md          run 3 (EMBER-2018): supersedes RESULTS.md where they differ
  figures_numerical.tex    LaTeX emitted from run 2's results/, no number typed by hand
  figures_numerical_ember.tex  the same, emitted from run 3 -- the figures the paper carries
tests/                     unit tests
```

---

## Installation

```bash
git clone https://github.com/<repository>.git t-xai
cd t-xai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # pinned to the recorded environment
```

`requirements.txt` pins the exact versions the published numbers were produced under
(Python 3.14.5, numpy 2.4.6, scikit-learn 1.8.0, shap 0.51.0, lime 0.2.0.1). For looser bounds —
using the code on new data rather than reproducing these results — install the package instead:

```bash
pip install -e ".[ablation,ember,test]"
```

`lief` is needed only on the EMBER-2018 path, and only because the vendored upstream extractor
imports it at module scope; no code path in this repository calls into it. See
`src/txai_exp/third_party/PROVENANCE.md`.

---

## Data

**Neither corpus is redistributed here.** Both are public, obtained from their publishers under
the publishers' own terms. Every load path in this repository is read-only.

| Corpus | Obtain from | Used for |
|---|---|---|
| **BODMAS** (134,435 × 2,381 static PE features, timestamped) | <https://whyisyoung.github.io/BODMAS/> | the primary instantiation, E1–E8 |
| **EMBER-2018** (1,000,000 × 2,381, feature version 2) | <https://github.com/elastic/ember> | the second-corpus replication, E9 |

Point one environment variable at wherever you put them:

```bash
export TXAI_DATA_DIR=/path/to/corpora
```

expected to contain:

```text
$TXAI_DATA_DIR/
  bodmas.npz                 # BODMAS features and labels
  bodmas_metadata.csv        # BODMAS timestamps, row-aligned with the above
  ember2018/                 # the released EMBER-2018 JSONL, extracted
  ember2018_matrix/          # written by build_ember_matrix.py (see below)
```

Nothing silently assumes a layout. `txai_exp.data.verify_feature_layout` re-derives the EMBER-v2
column order from an arithmetic invariant — columns 0:256 and 256:512 must each sum to exactly 1
in every row, which no other 256-wide block does — and raises if the file is not the release this
code assumes. A dataset or extractor-version mismatch fails loudly rather than producing plausible
wrong numbers.

---

## Reproducing the results

No GPU is required. The full programme runs in about an hour on an 8-core laptop with 16 GB of
RAM, excluding the optional EMBER-2018 replication; `docs/PROTOCOL.md` records the measured cost
of every stage. Each driver is independent, writes its output to `results/`, and resumes rather
than redoing finished work. Wall-clock figures below are the recorded ones.

```bash
python3 experiments/run_scoring_layer.py --stage all            # ~26 min  -> E1,E2,E5,E6,E7
python3 experiments/run_calibration_falsification.py            # ~5 min   -> E3
python3 experiments/run_self_stability.py                       # ~24 min  -> E4b
python3 experiments/run_kernel_constant.py                      # ~2 s     -> E5b
python3 experiments/run_release_game.py                         # ~6 min   -> E8
python3 experiments/make_figures.py                             # -> docs/figures_numerical.tex
```

Run 3 re-measures the whole numerical section on EMBER-2018 and adds the four measurements a
round-3 review asked for. It needs `TXAI_EMBER_MATRIX` pointed at a built matrix, and it is the
run the manuscript now reports; `docs/ROUND3_EMBER.md` records what it found.

```bash
python3 experiments/run_numerical_instantiation.py --corpus ember   # ~15 min -> section7_ember.json
python3 experiments/run_ember_ablations.py --stage all              # ~26 min -> ablations_ember.json
python3 experiments/run_release_game_ember.py                       # ~17 min -> E8_release_game_ember.json
python3 experiments/run_self_stability.py --corpus ember            # ~37 min -> E4b_self_stability_ember.json
python3 experiments/run_kernel_constant.py --source section7_ember  # ~seconds
python3 experiments/make_figures.py --source section7_ember \
    --bridge E5b_kernel_constant_ember > docs/figures_numerical_ember.tex
```

`--corpus bodmas` runs the identical code on the run-2 substrate, which is what makes the two
comparable.

The EMBER-2018 replication is a two-step path, and the first step is the expensive one — it
vectorises a million samples into a 8.9 GB memmap:

```bash
python3 experiments/build_ember_matrix.py \
    --src "$TXAI_DATA_DIR/ember2018" --out "$TXAI_DATA_DIR/ember2018_matrix" --workers 7
bash experiments/supervise_ember_transfer.sh 104578              # E9; retries, halves on OOM
bash experiments/watch_ember_transfer.sh                          # optional, read-only monitor
```

`build_ember_matrix.py` is chunked and resumable, and the loader refuses an incomplete matrix
rather than reading unwritten rows as real samples — an unfinished run is zeros, and zeros are
valid-looking feature vectors. `--limit-chunks 2` gives a smoke test in seconds.

### Tests

```bash
python3 -m pytest tests/ -q
```

**75 passed, 9 skipped** with no corpus present; the skips are the tests that need one, and they
run once `TXAI_DATA_DIR` is populated. The suite is written around the failures that would
otherwise be silent: a hashing convention that lands the vectors in a different feature space from
the corpus they are compared against, a resume that redoes or skips work, and a loader that hands
out a matrix with holes in it.

---

## Findings

Condensed below, **as measured in run 2 on BODMAS**. Full statements, evidence pointers, the
claim-to-evidence matrix, and the scope these numbers are admissible for are in
`docs/RESULTS.md` and `docs/EVIDENCE_GATE.md`.

> [!IMPORTANT]
> **F3 and F4 do not survive run 3.** Re-measured on EMBER-2018 under nine instantiations of
> the definitions, F3's margin is 2 grid points of 5,324 and its sign reverses in 7 of the 9;
> and strict calibration rejects **every** map, TreeSHAP included, rather than only the
> control. `docs/ROUND3_EMBER.md` gives both in full. The other six findings stand.

| | Finding | Evidence |
|---|---|---|
| **F1** | The Theorem 1 bound holds in **138 / 138** measured configurations and is never tight — median LHS/RHS 0.029 for TreeSHAP, max 0.243. Verified and quantifiably loose; not sharp. | E5 |
| **F2** | About half the slack is the constant, not the kernel: the kernel's Dobrushin coefficient in place of $L_\pi = 1$ tightens the median ratio to 0.064 (factor 2.19) with zero violations. | E5b |
| **F3** ⚠️ | The numeric triple $\{F, B, D\}$ **does not separate an explanation from a constant vector**. At $(\tau_f, \tau_b, \tau_d) = (0.5, 0.8, 0.8)$ the constant control is admissible on 100 % of alerts against TreeSHAP's 94.4 %, and over a larger share of the threshold surface (28.1 % vs 22.5 %). | E7 |
| **F4** ⚠️ | Equation (7)'s calibration condition **is** the filter the thresholds are not: the constant control violates it on 200 / 200 alerts at every $k$. It also bites on TreeSHAP — 62 / 200 strict violations at $k = 50$. | E3 |
| **F5** | LIME violates Equation (6)'s stated precondition (views deterministic conditional on $x$): self-stability 0.590 with zero identical repeats, against 0.595 under a 1 % input perturbation. Sixteen times the neighbourhood samples does not fix it. | E4b |
| **F6** | Equation (10)'s optional $[0,1]$ normalization destroys the role ordering it exists to make comparable — non-monotone on 382 / 600 role pairs, and flat — while raw $D$ is monotone in privilege with zero violations. | E6 |
| **F7** | BODMAS separability is a collection artefact specific to that corpus. Eight of nine EMBER feature groups reach AUC ≥ 0.986 alone on BODMAS; on EMBER-2018, same detector, same training size (104,578), **none of the nine does** — best is `sections` at 0.931, against a full-feature AUC of 0.988. | E1, E9 |
| **F8** | The release game is solvable once its payoffs carry measured quantities. Release width is nondecreasing in the evidence weight $\eta$, nonincreasing in the disclosure weight $\rho$, and sits at the lattice minimum for every $\rho$ when $\eta = 0$. The equilibrium is pure at all 18 settings. | E8 |

---

## Provenance and licence

Code in this repository is MIT-licensed (`LICENSE`). One file is vendored from upstream —
`src/txai_exp/third_party/ember_features.py`, MIT, byte-identical, with its SHA-256 recorded so the
claim is checkable:

```bash
shasum -a 256 src/txai_exp/third_party/ember_features.py
# db0d93bb1e1d4b28558e373c77194a82c3cae09d7a1aeb44eea5dd68e84f3ffd
```

`src/txai_exp/third_party/PROVENANCE.md` records why it is vendored rather than installed, and why the
LIEF version warning it prints does not apply to the path this code uses. Dataset terms are the
publishers'; see `THIRD_PARTY_LICENSES.md`.

## Citation

The manuscript is under review and its authorship is withheld for that reason. Please cite the
repository as an anonymous artifact for now — `CITATION.cff` carries a machine-readable form that
will be updated when review concludes.
