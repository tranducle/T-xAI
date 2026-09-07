# T-XAI experiment results — run 3, 2026-08-04: Section VII on EMBER-2018

This run re-measures the manuscript's numerical section on **EMBER-2018** instead of
BODMAS and adds four follow-up measurements. It is the primary numerical record for the
current manuscript. `RESULTS.md` is retained as the dated BODMAS record. Current claim
mapping and theorem-scope rules are summarized in `MANUSCRIPT_ALIGNMENT.md`.

Raw outputs: `results/*_ember.json`. Every number below is transcribed from those files.

> **Current follow-up.** The manuscript now also reports E10 (`results/E10_delta_z_bridge.json`)
> and E11 (`results/E11_m3_evidence_channel.json`). E10 supplies the separate, within-scope
> explanation-only `Delta_z` instantiation that the M1 rows below cannot provide. E11 supplies a
> controlled authenticated M3 evidence-channel test. See `ACCEPTANCE_VALIDATION.md` and the
> scientific-gate records under `results/gates/` for their bounded claim permissions.

| | |
|---|---|
| Machine | 8-core arm64 laptop, 16 GB RAM, **no GPU** |
| Environment | Python 3.14.5, numpy 2.4.6, scikit-learn 1.8.0, shap 0.51.0, lime 0.2.0.1, xgboost 3.2.0 |
| Total wall-clock | **94.2 min** across the five drivers below |

> **On the identifiers.** As in `RESULTS.md`, the `E`- and `F`-codes are this repository's
> and do not appear in the manuscript; the `C`-codes are the join. The driver
> `run_numerical_instantiation.py` writes `results/section7_ember.json` — the filename
> keeps the section number because it is the name the manuscript's revision record cites.

---

## 1. Why the corpus changed

BODMAS is separable enough that the detector's alert population contains **no benign
samples at all**. Three consequences, and all three are properties of the corpus rather
than of the framework:

- the latent world `H` is constant, so the joint law over (world, action) equals its own
  action marginal, and the propositions that are *about* that distinction cannot be
  exercised on it;
- eight of nine EMBER feature groups separate BODMAS on their own, so every faithfulness,
  calibration and robustness number describes an explanation problem with almost nothing
  in it;
- the headline finding of run 2 was measured under a single disclosure cost and a single
  faithfulness score, and a finding read off one instantiation generalises only as far as
  that instantiation does.

On EMBER-2018, at the same training size (104,578) and the same detector, the detector
reaches **AUC 0.988434** against BODMAS's 0.99994, **none** of the nine feature groups
reaches AUC 0.986 alone (best: `sections` at 0.9305), and the alert population carries
**63 benign alerts of 1,000** — world mass `benign 0.085 / malware 0.915`. The distinction
is measurable, and it was not before.

## 2. What did not survive

**The run-2 headline finding is retracted in its general form.** Run 2 reported that the
numeric triple `{F, B, D}` does not separate an explanation from a constant vector, on the
strength of the constant control covering more of the threshold surface than TreeSHAP.
Re-measured on EMBER-2018 under **nine** instantiations — three faithfulness scores
(`deletion_auc`, `random_relative`, `flip_point`) × three disclosure costs
(`weighted_cover`, `unweighted_count`, `mass_weighted`) — the margin is:

| F | D | constant | TreeSHAP | constant wider? |
|---|---|---|---|---|
| `deletion_auc` | `weighted_cover` | 0.21074 | 0.21037 | **yes** |
| `deletion_auc` | `unweighted_count` | 0.24793 | 0.21037 | **yes** |
| `deletion_auc` | `mass_weighted` | 0.09917 | 0.37190 | no |
| `random_relative` | `weighted_cover` | 0.03512 | 0.21037 | no |
| `random_relative` | `unweighted_count` | 0.04132 | 0.21037 | no |
| `random_relative` | `mass_weighted` | 0.01653 | 0.37190 | no |
| `flip_point` | `weighted_cover` | 0.03512 | 0.14726 | no |
| `flip_point` | `unweighted_count` | 0.04132 | 0.14726 | no |
| `flip_point` | `mass_weighted` | 0.01653 | 0.25995 | no |

`finding_survives_all_instantiations: false`. Under the original (F, D) pair the constant
control leads by **0.00038 of the threshold surface — 2 grid points of 5,324** — and the
sign **reverses in 7 of the 9** instantiations. What generalises is not "the numeric
conditions admit a constant map"; it is that *this* pair of definitions does, and that a
per-alert random baseline collapses the control immediately.

Evidence: `section7_ember.json` → `addons.sensitivity`.

## 3. What the contract certifies here: nothing, at strict calibration

Sweeping the sufficiency threshold from `Φ = 1` to `Φ ≥ θ` gives a **calibration frontier**
— the largest θ each map can actually promise at a given release width. Measured:

| map | θ\* at k = 5, 10, 20, 50 |
|---|---|
| TreeSHAP | 0.0, 0.0, 0.0, 0.0 |
| LIME | 0.0, 0.0, 0.0, 0.0 |
| random | 0.0, 0.0, 0.0, 0.0 |
| constant | 0.0, 0.0, 0.0, 0.0 |

The worst-case frontier collapses to zero for **every** map at **every** width, because a
single alert with `Φ = 0` is enough to take it there. Strict calibration therefore voids
every admissibility verdict issued at that threshold — TreeSHAP included — and relaxing the
threshold does not repair it. The unit that carries information is the pair
**(θ, coverage)**, not θ alone.

Evidence: `section7_ember.json` → `addons.calibration_sweep`.

## 4. Positive controls: every check that reported zero now has a known defect it catches

Three checks in run 2 reported zero violations. A check that has never failed has not been
tested, so each is now given a defect it should notice:

| Check | Injected defect | Fired? |
|---|---|---|
| View monotonicity (Prop. `lp`) | `w_R(strings)` set to −1.0 for every role | **yes**, 415 violations |
| Cross-role cost ordering | analyst and auditor weight columns transposed | **yes**, 500 / 500 |
| Faithfulness separates an ordering from its reverse | attribution order reversed | **yes**, F 0.9956 → 0.0533 |
| Kernel column-stochasticity | column sums set to 1.5 | **yes**, caught |
| Bridge diagnostic | *falsification margin* rather than a defect | see below |

For the recorded `M1` bridge diagnostic, measured instability would have to inflate by a
factor of **11.27** at the default configuration, and **4.19** at the configuration where
the diagnostic inequality is tightest, before that numerical inequality is violated. Closed
form and grid scan agree. These rows alter detector inputs and therefore lie outside the
formal theorem's explanation-side transformation scope. They are implementation and response-
sensitivity diagnostics, not an empirical theorem test.

**The lattice control found the manuscript wrong, not the code.** Run under the
manuscript's original Equation (31) ordering (`public ≺ auditor ≺ analyst ≺ designer`), the
check fires on **500 / 500** alerts at the `auditor ≥ analyst` pair. The measured mean raw
disclosure costs order the roles the way the code does:

```
public 4.0342   analyst 2.1718   auditor 1.3102   designer 0.4630
```

Equation (31) was corrected in the manuscript to match. The ordering is a policy choice, and
what this repository checks is that the lattice and the `w_R` table agree with each other.

Evidence: `section7_ember.json` → `addons.controls`.

## 5. The fourth admissibility condition

`A_Γ` — actionability — was uncomputed in run 2, and it is the one condition with a
principled reason to exclude a constant attribution. It is now instantiated as the product
of three factors in [0, 1]: decisiveness, playbook grounding, and stability under
perturbation. Measured means over the alert population:

| map | `A_mean` | `A_median` | decisiveness | grounding | stability |
|---|---|---|---|---|---|
| TreeSHAP | 0.1151 | 0.0793 | 0.2417 | 0.6186 | — |
| constant | 0.0301 | 0.0000 | 0.6057 | 0.0497 | 1.000 |
| random | 0.0281 | 0.0000 | 0.4722 | 0.0615 | 0.986 |
| LIME | 0.0187 | 0.0000 | 0.2336 | 0.0815 | 0.626 |

It does separate the maps: the controls are decisive and ungrounded, which is exactly the
failure mode a constant vector has. `τ_a` is swept rather than fixed and is deliberately
kept out of the main threshold grid, because the playbook it rests on is **illustrative**,
not a validated operational metric. Read `A_Γ` as a proxy.

Evidence: `section7_ember.json` → `addons.actionability`.

## 6. The joint/marginal distinction is now measurable

With a two-valued world, the release game's decision loss can be computed both ways. Across
**all 195 payoff cells** the marginal formula differs from the joint-law value — 195 / 195
nonzero, median gap **0.009559**. The equilibrium is pure (`k=20, softmax_t1` against
`M4_observe`), and `run_release_game_ember.py` reproduces the equilibrium of
`run_numerical_instantiation.py` to every digit on the same substrate.

Evidence: `E8_release_game_ember.json` → `measured.cells`, `equilibrium`.

## 7. `B` is an estimate, not a bound

`B` is a supremum estimated as the maximum over a sampled perturbation budget. A sampled
maximum is a **lower** estimate of a supremum, so the reported `B` is an **upper** estimate
of the true `B`, and it is budget-dependent. Theorem 1's almost-sure hypothesis is therefore
checked against an estimate rather than verified. A guaranteed upper bound is open work.

## 8. Status of the run-2 findings

| | Status after run 3 |
|---|---|
| **F1** Bridge diagnostic remains below its right-side expression | Stands numerically; scope corrected. It is an `M1` diagnostic, not theorem validation (§4) |
| **F2** Dobrushin coefficient tightens the constant | Stands; re-measured on EMBER in `E5b_kernel_constant_ember.json` |
| **F3** `{F, B, D}` does not separate a constant map | **Superseded — see §2.** True of one (F, D) pair, reversed in 7 of 9 |
| **F4** Calibration is the filter the thresholds are not | **Superseded — see §3.** Strict calibration rejects every map, TreeSHAP included |
| **F5** LIME violates the determinism precondition | Stands; re-measured in `E4b_self_stability_ember.json` |
| **F6** Normalised `D` destroys the role ordering | Stands, and §4 adds the lattice correction |
| **F7** BODMAS separability is a collection artefact | Stands, and is now the reason the whole section moved corpus |
| **F8** The release game is solvable | Stands, and §6 adds the joint/marginal gap it could not exhibit before |

## 9. Reproducing run 3

`TXAI_EMBER_MATRIX` must point at the matrix written by `build_ember_matrix.py`.

```bash
python3 experiments/run_numerical_instantiation.py --corpus ember   # 15.4 min -> section7_ember.json
python3 experiments/run_ember_ablations.py --stage all              # 25.6 min -> ablations_ember.json
python3 experiments/run_release_game_ember.py                       # 16.5 min -> E8_release_game_ember.json
python3 experiments/run_self_stability.py --corpus ember            # 36.7 min -> E4b_self_stability_ember.json
python3 experiments/run_kernel_constant.py --source section7_ember  # seconds  -> E5b_kernel_constant_ember.json
```

```bash
python3 experiments/make_figures.py --source section7_ember \
    --bridge E5b_kernel_constant_ember > docs/figures_numerical_ember.tex
```

`run_numerical_instantiation.py --corpus bodmas` runs the identical code on the run-2
substrate, which is what makes the two comparable. `docs/figures_numerical_ember.tex` is the
emitted output checked in here: every coordinate in the manuscript's Section VII figures is
transcribed from `results/*.json` by that command, and none is typed by hand.

## 10. Known limits of this run

- **The test suite does not cover the six modules added here** (`supplementary`, `controls`,
  `alt_metrics`, `actionability`, `calibration_sweep`, `ember/substrate`). `tests/` is
  unchanged from run 2. Their outputs were checked against the manuscript by hand, not by
  assertion.
- **`A_Γ` rests on an illustrative playbook** (§5) and is a proxy, not a validated metric.
- **The admissible fraction is unstable across ablations** — 0.002–0.010 over five seeds,
  and 0.102 under a random-forest detector. Quote it only with its seed and detector.
- `PROTOCOL.md` and `EVIDENCE_GATE.md` are run-2 documents. Their gate verdict (**PASS WITH
  CLAIM NARROWING**) was not re-derived for this run; §2 and §3 narrow two claims further
  than that verdict did.
