# T-XAI experiment programme — costed on this machine, no GPU

> [!NOTE]
> **This is the run-2 programme.** Run 3 (2026-08-04) re-ran the numerical section on
> EMBER-2018 and added four measurements; its costs and drivers are in `ROUND3_EMBER.md`.
> The hardware conclusion below is unchanged — run 3 also used no GPU, at 94.2 min total.


Date 2026-08-02. Gate verdict: `EVIDENCE_GATE.md` → **PASS WITH CLAIM NARROWING**.

## Hardware and the GPU question

| | |
|---|---|
| Machine | 8-core arm64 laptop, **16 GB RAM, no GPU** |
| Present | scikit-learn 1.8.0, **xgboost 3.2.0**, shap 0.51.0, lime, numpy 2.4.6, scipy 1.17.1, pandas, matplotlib |
| Absent | torch, captum, cvxpy, nashpy |

**No GPU is required, and not marginally so.** The figures below are measured, not
estimated -- recorded output of a short CPU benchmark run on the machine above. The
benchmark harness is a scratch script and is not part of this release; every cost that a
claim depends on is re-measured by the drivers in `experiments/` and recorded in
`results/`:

| Operation | Measured cost | Peak RSS |
|---|---|---|
| Load BODMAS (134,435 × 2,381 float32) | 1.5 s | 1.9 GB of 16 GB |
| Train HistGradientBoosting, 200 iters, 107k train rows | **114 s** | — |
| SHAP TreeExplainer, 200 alerts | **0.24 s** → **1.2 ms/alert** | — |
| $B$ (Eq. 6), 50 alerts × 20 perturbations, SHAP | **1.08 s** | — |
| LIME, 5,000-sample neighbourhood | **0.65 s/instance** | — |

Why there is nothing for a GPU to do: the framework needs no deep learning. BODMAS is **tabular**
(2,381 static PE features), the detector is a **tree ensemble**, and TreeSHAP is an exact
combinatorial algorithm over tree paths — it is memory-bound integer work, not dense linear
algebra. A GPU accelerates dense matrix multiplication; none of the above is that.

Projected cost of the whole Tier 1 + Tier 2 programme below: **under 3 hours wall-clock** on this
laptop, dominated by LIME.

### Where a GPU *could* matter, stated honestly

1. **A neural detector** (MalConv or a deep MLP) as a second substrate, explained with
   Integrated Gradients via `captum`. This needs `torch`, which is absent. Even then: an MLP over
   2,381 tabular features trains on M2 CPU in minutes, and PyTorch on Apple Silicon uses the MPS
   backend regardless — so this is a *dependency* decision, not a hardware one.
2. **KernelSHAP instead of TreeSHAP** — model-agnostic and ~1000× slower. Unnecessary: the
   detector is a tree, so the exact algorithm applies.
3. **Very large $|\Delta|$ in Eq. (6)'s supremum.** This is the one genuinely superlinear cost:
   $K$ perturbations means $K$ re-explanations per alert. At SHAP's 1.2 ms/alert it stays cheap
   even at $K=1000$; at LIME's 0.65 s/alert it is the binding constraint (see E4).

**Conclusion: buy no hardware. Nothing in Tier 1–3 needs it.**

## The experiments

Numbering matches the gate matrix. Each states which contribution it serves and *in which of the
three legitimate senses* (non-vacuity / tightness / instantiability) — a theory paper's
experiments cannot "confirm" a proof.

> **On the identifiers.** `E1`–`E9` here, and the finding identifiers `F1`–`F8` in
> `RESULTS.md`, are labels for this repository only. **They do not appear in the manuscript**,
> which numbers its own material as contributions `C1`–`C5`, gaps `G1`–`G5`, and threat classes
> $M_1$–$M_4$. A reader holding both documents maps between them through the *contribution* each
> experiment names below — `E`-codes are the route back from a published number to the driver
> and the JSON that produced it, not a cross-reference into the paper.

### Tier 1 — must run before any manuscript claim

**E1 · Detector substrate and separability characterisation.**
Train HistGB and XGBoost under a **temporal** split (boundary 2020‑07‑08, 107,548 train /
26,887 test), ≥5 seeds. Report AUC, and the operating point at a realistic SOC false-positive
budget — AUC is the wrong headline metric for a detector whose alerts a human must triage.
Then characterise the AUC 0.9999 finding: attribution mass concentration, and how few features
suffice to recover the decision. *Serves C1 (instantiability). Cost ≈ 20 min for 10 fits.*

**E2 · $F_\Gamma$ faithfulness.** Deletion and insertion curves over SHAP and LIME attributions,
1,000 alerts. *Serves C2 (instantiability). Cost: SHAP ~1 s, LIME ~11 min.*

**E3 · Falsify the Eq. (7) calibration condition.** The condition requires
$F_\Gamma(x)\ge\tau_f \Rightarrow \Phi(z,x)=1$. Instantiate $\Phi$ as an *independent* sufficiency
test (not the one used to compute $F$ — otherwise it is circular), sweep $\tau_f$, count
violations. *Serves C2 (non-vacuity) — this is the experiment that decides whether the paper's
central precondition is a real filter or a formality.* Either outcome helps: frequent violations
mean the condition rejects current practice; rare violations mean it is cheap to meet.
*Cost: reuses E2, minutes.*

**E4 · $B_{\Gamma,r}$ robustness under $M_1$.** Functionality-preserving PE perturbations
(section padding, import addition — features only grow), $K$ perturbations per alert, SHAP and
LIME. Today's indicative run gave mean $B=0.63$, min $0.00$ at $K=20$ — i.e. **some explanations
collapse entirely under a 1 % feature perturbation**, which is the interesting direction.
*Serves C2 and C3/$M_1$ (non-vacuity).*
*Cost: SHAP, 5,000 alerts × 50 perturbations ≈ 3 min. LIME, 500 alerts × 20 ≈ **1.9 h** — the
programme's single largest cost, and the reason to keep LIME's alert count small.*

**E5 · Theorem 1 bridge tightness — the flagship.**
Theorem 1 states
$\mathrm{Adv}^{\mathrm{TV}}_{E,\Gamma}(\sigma;\Delta_z)\le\min\{1, L_\pi\,\mathbb{E}_x[1-B^z_{\Gamma,r}(x)]\}$.
Both sides are computable here:

- **RHS** — $\mathbb{E}[1-B^z]$ comes straight from E4 (today: $0.37$ at $K=20$). $L_\pi$ is *known
  by construction*, because we choose the response kernel $\pi$: a softmax over a finite action
  set with a stated temperature has an analytically known Lipschitz constant.
- **LHS** — simulate $\pi$ over the finite action set $\mathcal{U}$, estimate the joint
  $(\text{world},\text{action})$ pmf with and without $\delta$, take the TV distance, maximise
  over $\Delta_z$.

Report the **ratio LHS/RHS**. *Serves C5 (tightness) and simultaneously closes open thread G‑1,
which has stood as "no measured $\mathrm{Adv}^{TV}$".* A bound that is 100× slack should be said
to be 100× slack; that is a finding, not a failure. *Cost: reuses E4, minutes.*

### Tier 2 — Q1-strengthening, appendix-safe

**E6 · $D_\Gamma$ disclosure and least-privilege dominance.** Compute the constructive witness
$D_\Gamma(z,R)=\sum_{c\in\mathrm{comp}(z)}w_R(c)$ over top-$k$ attributed components per role, and
verify numerically that the ordering Prop. `lp` predicts actually holds on real attributions.
*Serves C2 + C5 (instantiability). Cost: seconds.*

**E7 · Admissibility rate (Def. 3).** Fraction of alerts meeting
$F\ge\tau_f,\;B\ge\tau_b,\;D\le\tau_d$ — **over three conditions, not four**: $A_\Gamma$ is
excluded because BODMAS supports no response playbook, and inventing one would fabricate the
governance layer the paper is about. This turns Table IV's `Status` column from illustrative into
measured, with the $A$ column explicitly marked assumed. *Serves C2 (instantiability). Cost: seconds.*

### Tier 3 — requires an author decision, because it changes declared scope

**E8 · Compute a strong Stackelberg equilibrium on a small release game.**
Feasible without `cvxpy`: `scipy.optimize.milp` (HiGHS, present in scipy 1.17) solves the standard
MILP formulation for small finite games. **But the manuscript currently states that computing
equilibria is outside its scope (C4).** Running E8 would extend C4 rather than support it, and
that is a scope change only the author should make. Not started.

## What is out of reach, restated so it is not quietly forgotten

$\Omega$/provenance, the claim-evidence graph (Def. 4), the role-based release lattice with real
consequences, $A_\Gamma$/$\Psi$, and threat class $M_3$ **cannot be exercised on BODMAS**.
Instantiating them synthetically and reporting the result as validation would fabricate exactly
the governance layer this paper contributes. The write-up must say the programme exercises the
scoring layer and leaves the governance layer untested.

## Execution order

E1 → E2 → (E3 ∥ E4) → E5 → E6 → E7. E3 and E4 both consume E2's explanations and are independent
of each other. Total ≈ 2.5 h, one laptop, no GPU, no new dependency.
