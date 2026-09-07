# T-XAI: Threat-Model-Aware Explanation Release in Cybersecurity

> **Double-anonymous review artifact.** This repository intentionally contains no author names,
> affiliations, acknowledgments, funding statements, contact addresses, institutional paths, or
> identity-bearing repository URLs. Please preserve that constraint until peer review concludes.

This repository contains the reproducibility and measurement artifacts for T-XAI, a contract for
reasoning about whether an explanation may be released to a particular role and used as evidence
for a permitted cybersecurity response under a declared adversary model.

## Scope

The current manuscript separates the formal framework from what is empirically exercised.

**Measured here:**

- explanation faithfulness, robustness, disclosure, and an illustrative actionability proxy;
- strict calibration and certificate-abstention behavior;
- sampled `M1` input perturbations and explanation robustness diagnostics;
- response-kernel and bridge-related diagnostic quantities;
- disclosure monotonicity and deliberately defective controls;
- detector and random-seed ablations;
- one finite Stackelberg release-game instance under `M1` and `M4`.

**Not empirically established here:**

- `M2` detector or rule manipulation;
- `M3` explanation-channel or evidence-record manipulation;
- the full claim-evidence audit protocol or evidence-authenticity assumption;
- operationally validated actionability in a deployed response workflow;
- organization-calibrated release-game loss, capability-cost, and information-value terms;
- human analyst benefit;
- empirical validation of the robustness-to-decision theorem.

The last point is important. The available experiments perturb detector inputs under `M1`. The
formal theorem instead assumes explanation-side transformations that leave non-explanation inputs
fixed. Therefore the repository computes useful bridge-related diagnostics, but those rows are not
an empirical theorem test.

See `docs/MANUSCRIPT_ALIGNMENT.md` for the current claim-to-artifact map.

## Current numerical substrate

The manuscript now reports EMBER-2018 as its primary numerical substrate because it provides a
non-degenerate alert population for the joint world/action quantities used by the release model.
The earlier BODMAS measurements are retained as a dated diagnostic record and remain useful for
cross-corpus comparison, but they are not the headline manuscript evidence.

| Corpus | Role in the current artifact | Obtain from |
|---|---|---|
| **EMBER-2018** | Primary numerical instantiation reported in the manuscript | <https://github.com/elastic/ember> |
| **BODMAS** | Earlier diagnostic run and cross-corpus comparison | <https://whyisyoung.github.io/BODMAS/> |

Neither corpus is redistributed here.

## Repository layout

```text
src/txai_exp/              measurement package
experiments/               experiment drivers
results/                   frozen JSON outputs used by the numerical study
docs/
  MANUSCRIPT_ALIGNMENT.md  current manuscript-to-artifact mapping and evidence boundaries
  ROUND3_EMBER.md          current EMBER-2018 numerical record
  RESULTS.md               earlier BODMAS record, retained for provenance
  PROTOCOL.md              experiment protocol record
  EVIDENCE_GATE.md         evidence gate and claim-narrowing record
  figures_numerical_ember.tex
                           generated numerical figure/table source for the current run
tests/                     unit tests
```

## Installation

```bash
git clone https://github.com/<anonymous-repository>.git t-xai
cd t-xai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The frozen environment records Python 3.14.5, NumPy 2.4.6, scikit-learn 1.8.0,
SHAP 0.51.0, LIME 0.2.0.1, and XGBoost 3.2.0.

For editable installation on new data:

```bash
pip install -e ".[ablation,ember,test]"
```

## Data layout

Point `TXAI_DATA_DIR` at the local corpus directory:

```bash
export TXAI_DATA_DIR=/path/to/corpora
```

Expected layout:

```text
$TXAI_DATA_DIR/
  bodmas.npz
  bodmas_metadata.csv
  ember2018/
  ember2018_matrix/
```

The EMBER matrix is created locally from the released JSONL data and is not committed.

## Reproducing the manuscript-aligned EMBER run

```bash
python3 experiments/run_numerical_instantiation.py --corpus ember
python3 experiments/run_ember_ablations.py --stage all
python3 experiments/run_release_game_ember.py
python3 experiments/run_self_stability.py --corpus ember
python3 experiments/run_kernel_constant.py --source section7_ember
python3 experiments/make_figures.py --source section7_ember \
    --bridge E5b_kernel_constant_ember > docs/figures_numerical_ember.tex
```

The expensive preprocessing step is resumable:

```bash
python3 experiments/build_ember_matrix.py \
    --src "$TXAI_DATA_DIR/ember2018" \
    --out "$TXAI_DATA_DIR/ember2018_matrix" \
    --workers 7
```

## Manuscript-aligned evidence summary

The current manuscript uses the following bounded interpretations.

1. **Strict calibration binds.** None of the four tested explanation maps satisfies the strict
   certificate profile over all audited alerts. Under this profile T-XAI abstains from issuing a
   formal certificate. This is not a claim that every explanation from those maps is operationally
   useless.
2. **Score-map choice matters.** The constant-control comparison reverses across alternative
   faithfulness and disclosure instantiations. The result supports publishing the score
   instantiation with any admissibility verdict rather than treating one score formula as universal.
3. **Sampled `M1` robustness is diagnostic.** TreeSHAP robustness changes with perturbation budget
   and sample depth. LIME also carries explainer self-noise, so deterministic and stochastic maps
   are not interpreted identically.
4. **Bridge quantities remain numerically loose under `M1`.** Across the recorded diagnostic
   configurations the computed left-side quantity remains below the corresponding right-side
   expression. These rows are outside the theorem's explanation-side transformation scope and do
   not empirically validate the theorem.
5. **The validators can fail when they should.** Deliberately defective inputs trigger the relevant
   checks, including disclosure monotonicity, role ordering, faithfulness ordering, and kernel
   validity controls.
6. **The release game is solvable for one finite instance.** The reference solution is a pure
   equilibrium, but its utility model contains partly stipulated terms. The equilibrium width is
   therefore an instance result, not a deployment recommendation.
7. **Ablations preserve qualitative patterns, not one universal admissible fraction.** Detector and
   seed changes leave several qualitative observations intact while changing the quantitative
   admissibility surface.

For exact numbers, sample counts, and frozen output paths, see `docs/ROUND3_EMBER.md` and the JSON
files in `results/`.

## Tests

```bash
python3 -m pytest tests/ -q
```

Tests that require a local corpus are skipped when `TXAI_DATA_DIR` is not populated.

## Provenance and third-party code

The project code is MIT licensed. A vendored EMBER feature extractor is kept byte-identical to its
upstream source and has its SHA-256 recorded in `src/txai_exp/third_party/PROVENANCE.md`.
Dataset licenses remain those of the original publishers. See `THIRD_PARTY_LICENSES.md`.

## Citation during review

Please cite this repository as an anonymous research artifact while the manuscript is under
review. `CITATION.cff` intentionally withholds authorship and will be updated only after review.
