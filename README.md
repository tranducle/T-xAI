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
- a 500-alert, 18-transformation explanation-only `Delta_z` follow-up that instantiates the formal bridge scope;
- a controlled `M3` claim-evidence channel experiment with authenticated records and a chained producer/verifier log;
- an E12 BODMAS replication of the `Delta_z` and controlled-`M3` mechanisms using a different detector family and explanation mechanism;
- disclosure monotonicity and deliberately defective controls;
- detector and random-seed ablations;
- one finite Stackelberg release-game instance under `M1` and `M4`.

**Not empirically established here:**

- `M2` detector or rule manipulation;
- arbitrary or live-SOC `M3` compromise, including compromised signers or key-management infrastructure;
- operationally validated actionability in a deployed response workflow;
- organization-calibrated release-game loss, capability-cost, and information-value terms;
- human analyst benefit;
- cross-domain, cross-modality, or deployment generality.

The bridge evidence has two distinct scopes. The original `M1` experiments change detector inputs and remain diagnostics outside the formal theorem. E10 instead holds inputs, detector outputs, and labels exactly fixed while transforming only delivered explanations. E12 repeats that mechanism on BODMAS with a different detector and deterministic linear-contribution explainer. E10 and E12 are therefore numerical instantiations inside the theorem's declared `Delta_z` scope, not additional proof or a claim of general empirical validation.

See `docs/MANUSCRIPT_ALIGNMENT.md` for the current claim-to-artifact map.

## Current numerical substrate

The manuscript now reports EMBER-2018 as its primary numerical substrate because it provides a
non-degenerate alert population for the joint world/action quantities used by the release model.
BODMAS serves two roles: the earlier full-program run is retained as a dated diagnostic record, and E12 uses a temporally held-out subset for an external replication of the `Delta_z` and controlled-`M3` mechanisms with a different detector and explanation mechanism. The headline scoring, calibration, disclosure, and release-game evidence remains EMBER-2018.

| Corpus | Role in the current artifact | Obtain from |
|---|---|---|
| **EMBER-2018** | Primary numerical instantiation reported in the manuscript | <https://github.com/elastic/ember> |
| **BODMAS** | Earlier full-program diagnostic plus E12 external mechanism replication | <https://whyisyoung.github.io/BODMAS/> |

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
  ACCEPTANCE_VALIDATION.md E10/E11/E12 follow-up evidence and claim-permission summary
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

The acceptance-validation follow-ups are reproduced separately so their scope is explicit:

```bash
python3 experiments/run_acceptance_gates.py
python3 experiments/run_delta_z_bridge.py
python3 experiments/run_m3_evidence_channel.py
python3 experiments/run_external_replication.py
python3 experiments/run_acceptance_result_gates.py
```

E10 and E11 record Git revision `a43c96c192b3f0b475d887adcb494a75a9017359`; E12 records revision `55545e9d5b144492251702e0e8550c8baf411f51`. All official records require `working_tree_dirty=false`.

The expensive preprocessing step is resumable:

```bash
python3 experiments/build_ember_matrix.py \
    --src "$TXAI_DATA_DIR/ember2018" \
    --out "$TXAI_DATA_DIR/ember2018_matrix" \
    --workers 7
```

## Manuscript-aligned evidence summary

The current manuscript uses the following bounded interpretations.

1. **Strict calibration binds.** None of the four tested explanation maps satisfies the strict certificate profile over all audited alerts. T-XAI therefore abstains from issuing that formal certificate; this is not a claim that every individual explanation is useless.
2. **Score-map choice matters.** The constant-control comparison reverses across alternative faithfulness and disclosure instantiations, so score definitions are part of the auditable policy rather than neutral implementation details.
3. **Sampled `M1` robustness remains diagnostic.** TreeSHAP robustness changes with perturbation budget and sampling depth, and stochastic explainers carry self-noise. The associated bridge rows remain outside the theorem because they change detector inputs.
4. **E10 instantiates the theorem inside its declared `Delta_z` scope.** On 500 alerts, 18 deterministic explanation-only transformations leave inputs, detector scores, and labels exactly unchanged. All 12 family/kernel rows satisfy the implemented generic and Dobrushin inequalities; the largest generic LHS/RHS ratio is 0.613. This is an executable within-scope instantiation, not proof or general validation.
5. **E11 exercises controlled `M3` integrity.** Authenticated provenance rejects all seven tested bundle corruptions with zero observed clean false rejection, while structural checks miss all six well-formed edits. A chained producer/verifier channel detects edit, insertion, truncation, replay, reorder, rollback, and tag corruption, while accepting canonical reserialization and legitimate append. Full retagging succeeds when the authenticator key and terminal anchor are compromised, which explicitly preserves the authenticity boundary.
6. **E12 reproduces both targeted mechanisms under a different stack.** On a temporal BODMAS holdout, an SGD-logistic detector and deterministic linear-contribution explainer reproduce all 12 within-scope bridge rows and the controlled authenticated-channel behavior. This reduces dependence on one corpus/detector/explainer stack, but both corpora remain static PE malware.
7. **The release game is solvable for one finite instance.** Its equilibrium is an instance result because several utility terms remain stipulated rather than organization-calibrated.
8. **Ablations preserve qualitative patterns, not one universal admissible fraction.** Detector and seed changes leave several observations intact while changing the quantitative admissibility surface.

Exact E10/E11/E12 numbers and claim boundaries are recorded in `docs/ACCEPTANCE_VALIDATION.md`, `results/E10_delta_z_bridge.json`, `results/E11_m3_evidence_channel.json`, `results/E12_external_replication.json`, and `results/gates/`.

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
