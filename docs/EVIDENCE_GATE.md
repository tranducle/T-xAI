# SCIE Q1 Evidence Coverage Gate — T-XAI experiment programme

> [!CAUTION]
> This is a dated pre-current-manuscript evidence-gate record. Contribution and equation
> numbering in this file may reflect an earlier draft. Use
> `MANUSCRIPT_ALIGNMENT.md` for the current C1-C4 mapping and bridge-scope rule.
> Recorded `M1` bridge rows are diagnostics and do not empirically validate the
> robustness-to-decision theorem.


Run 2026-08-02, before any experiment was batched, as the project's evidence protocol
requires: no claim enters the manuscript that the planned experiments cannot support.
Inputs: the manuscript's contributions C1–C5, a measured feasibility record, and two CPU
benchmarks run on the machine described in `PROTOCOL.md`.

## Verdict

**PASS WITH CLAIM NARROWING.**

Experiments may proceed. They exercise the framework's **scoring** layer (F, B, D, the Eq. (7)
calibration condition, and the Theorem 1 bound). They do **not** touch the **governance** layer
(Ω provenance, the claim-evidence graph, the role-based release lattice with real consequences,
and the $M_3$ adversary). Any manuscript sentence implying the governance layer was validated is
blocked until that evidence exists.

## What experiments can and cannot do for a theory paper

This is the framing that decides the whole programme, so it is stated before the matrix.

C1 (contract) and C4 (release game) are **definitions and an existence proof**. No experiment
confirms a definition, and no experiment confirms an existence theorem — a proof already settles
it. C5's statements are proved; running data through them cannot make them more true.

Experiments here therefore serve exactly three purposes, and every entry below is one of them:

1. **Non-vacuity** — the framework's conditions are satisfiable, and *not trivially* satisfiable.
   A definition nothing violates is decoration; a definition nothing satisfies is unusable.
2. **Tightness** — a proved bound is worth little if it is always slack. Measuring both sides of
   Theorem 1 says whether the bound bites.
3. **Instantiability** — every slot of the contract can be filled with a real value on real data,
   or it cannot, and the paper must say which.

"Beating a baseline" is not among them and must not appear in the write-up.

## Claim-to-Experiment Matrix

| Claim | Evidence required for Q1 | Planned experiment | Coverage | Missing | Downgrade if missing |
|---|---|---|---|---|---|
| **C1** Contract separates detector / method / audience / adversary / evidence / disclosure / action | Demonstration that each slot takes a real value on a real deployment | **E1+E7** fill 5 of 7 slots on BODMAS | **Partial** | Evidence obligation (Ω) and audience roles have no real referent in BODMAS | Say "instantiated on a static-malware detector; the provenance and role slots are specified but not exercised" |
| **C2** Risk functional + admissibility conditions | F, A, B, D computed on real alerts; admissibility rate reported; conditions shown non-vacuous | **E2, E3, E4, E6, E7** | **Good, except A** | $A_\Gamma$ (actionability) needs a response playbook that BODMAS has no basis for | Report admissibility over {F,B,D} only, and label $A$ as assumed, not measured |
| **C3** Four threat classes + audit protocol | Each class exercised, or its absence declared | **E4** exercises $M_1$ (input-only) end to end | **Partial by design** | $M_2$, $M_3$, $M_4$ not exercisable here | Already stated in the manuscript's limitations; keep it and cite the experiment as covering $M_1$ only |
| **C4** Stackelberg release game, existence of SSE | Nothing — existence is proved. Optional: a computed equilibrium on a small instance | **E8 (Tier 3)** | **N/A** | — | None. The paper already scopes equilibrium computation out; E8 would *extend* C4, not support it |
| **C5** One theorem + eight propositions | Bounds shown non-vacuous and their slack quantified | **E5** measures both sides of Theorem 1; **E6** checks Prop. `lp` dominance numerically | **Good** | Props `mono`, `suff`, `local`, `cover`, `sse`, `marginal` are definitional or negative results with nothing to measure | None — say plainly that E5 quantifies the bridge bound and the rest are proved, not tested |

## Coverage Audit

| Dimension | Status | Reviewer risk | Required action |
|---|---|---|---|
| Dataset breadth | **One dataset (BODMAS)** | High for CCS/TDSC, low for CSF | Title/abstract/contributions must not generalise past a single static-PE benchmark |
| Split discipline | **Measured today: temporal split gives AUC 0.9999, identical to random** | **High** | See "Threat to validity" below — must be reported, not hidden |
| Baselines | Two detectors (HistGB, XGBoost), both tree | Medium | Do not claim anything about detector families; the detector is a substrate here, not a contribution |
| XAI method coverage | SHAP + LIME | Low–medium | Limit explanation claims to these two; no claim about XAI methods as a class |
| Ablations | Threshold sweeps over $\tau_f,\tau_b,\tau_d$ | Adequate | Report the sweep, not a single tuned threshold |
| Statistics | Multi-seed required (detector training is stochastic) | Medium | ≥5 seeds; report spread, never a bare point estimate |
| Human study | **None** | High if claimed | No claim of analyst trust, usability, or human-validated explainability |
| Robustness | $M_1$ perturbations only | Declared | Frame as input-only; do not say "robust to explanation-aware attacks" |
| Reproducibility | Seeds, configs, scripts under this root | Good | Keep the benchmark scripts with the results |

## Threat to validity discovered while gating

The detector reaches **AUC 0.9999 under a temporal split** (train ≤ 2020‑07‑08, test after),
which is the same number a random split gives. I had hypothesised random-split leakage; the
temporal measurement **refutes** that hypothesis. The remaining explanation is that BODMAS's
benign and malicious samples are separable from how they were collected, and a temporal split
cannot remove a collection artefact.

Consequence for this programme: explanations of a near-perfectly separable detector may be
unrepresentatively stable or unrepresentatively concentrated. This does **not** invalidate E3 or
E5 — both ask questions about the *explanation method and the bound*, not about detector quality
— but it must be stated as a threat to validity in any write-up, and E1 must characterise it
rather than leave it as a suspicious number.

## Tiered decisions

- **Tier 1 must-run** (before any manuscript claim): E1, E2, E3, E4, E5.
- **Tier 2 Q1-strengthening** (appendix if space is tight): E6, E7.
- **Tier 3 future work** (changes the paper's declared scope — author decision required): E8.

## Separation of duties

SEOS requires that whoever validates the data is not whoever writes the resulting claims. The
experiment scripts and their JSON outputs are the hand-off artefact: claims must be written from
the recorded numbers, not from this session's narrative.

## Provenance note

BODMAS is a third-party corpus, obtained from its publishers and used read-only; nothing
here modifies or redistributes it. See the repository README ("Data") for the retrieval
link and the terms it is used under. Every number in this document is derived from that
corpus as released, not from a locally curated variant.
