# T-XAI experiment results — run 2, 2026-08-02

> [!IMPORTANT]
> **Restated in part by `ROUND3_EMBER.md` (run 3, 2026-08-04).** The manuscript's numerical
> section was re-measured on EMBER-2018 after a round-3 review. **F3** below is narrowed to
> the $(F, D)$ pair it was measured under — across three instantiations of each, the control
> is ahead in 2 of the 9 — and **F4** comes out stronger than it is stated here: strict
> calibration rejects all four maps, TreeSHAP included. The rest stand. This file is kept as
> the dated record of run 2 and has not been rewritten.


Programme: `PROTOCOL.md`. Gate and claim matrix: `EVIDENCE_GATE.md` (verdict **PASS WITH CLAIM
NARROWING**). Raw outputs: `results/*.json`. Code: `src/txai_exp/`, drivers `run_scoring_layer.py`,
`run_self_stability.py`, `run_kernel_constant.py`, `run_release_game.py`. Unit tests: `tests/` — **67
passed** (53 + 14 for E8).

Machine: 8-core arm64 laptop, 16 GB RAM, **no GPU**. Python 3.14.5, numpy 2.4.6, scikit-learn
1.8.0, shap 0.51.0, lime 0.2.0.1, xgboost 3.2.0 (`results/all_all.json` → `environment`).
Wall-clock: main programme **26.2 min**, self-stability 24 min, kernel analysis 2 s, E8 release
game 6.1 min.

> **On the identifiers.** The finding codes `F1`–`F8` below, and the experiment codes `E1`–`E9`
> they cite, belong to this repository and **do not appear in the manuscript**. The contribution
> codes `C1`–`C5` do: they are the manuscript's own numbering, and they are the join between the
> two documents. Read an `F`-code as "this repository's finding", never as a paper section.

> **What these experiments can do.** C1 and C4 are definitions and an existence proof; C5's
> statements are proved. No experiment confirms any of them; E8 solves one finite instance of C4's
> game, which is a tractability demonstration and not a confirmation of the proposition.
> Per `EVIDENCE_GATE.md` §"What
> experiments can and cannot do", the only admissible purposes are **non-vacuity**, **tightness**,
> and **instantiability**. Nothing below is a claim to beat a baseline, and no result here should
> be written up as one.

---

## 1. Headline findings

| # | Finding | Evidence | Consequence for the manuscript |
|---|---|---|---|
| **F1** | Theorem 1 holds in **138 / 138** configurations and is **never** tight: median LHS/RHS = **0.029** for TreeSHAP (~34× slack), max 0.243. | E5 | Report the bound as verified and quantifiably loose. Do not describe it as sharp. |
| **F2** | Roughly half of the temperature-attributable slack is the *constant*, not the kernel: replacing $L_\pi=1$ with the kernel's Dobrushin coefficient tightens the median ratio to **0.064** (factor **2.19**) with **0** violations. | E5b | A tighter constant is available constructively. Either adopt it or say why not. |
| **F3** | Definition 3's numeric triple $\{F,B,D\}$ **does not separate an explanation from a constant vector**: at $(\tau_f,\tau_b,\tau_d)=(0.5,0.8,0.8)$ the constant control is admissible on **100 %** of alerts vs TreeSHAP's **94.4 %**, and it is admissible over a *larger* share of the threshold surface (28.1 % vs 22.5 %). | E7 | The thresholds alone are not a filter. This is an argument *for* the framework's structure, not against it — see F4. |
| **F4** | Eq. (7) calibration **is** that filter: the constant control violates it on **200 / 200** alerts at every $k$, under either reading of $\Phi$. **No** explainer satisfies it strictly ($\Phi=1$ under every donor) on all alerts at any $k$ tested; TreeSHAP comes closest at $k=50$ — **62 / 200** strict violations, **1 / 200** total failures. | E3 | The manuscript's own words at `paper.tex:524` call such an instantiation *defective*. Eq. (7) is a real, falsifiable constraint that rejects the degenerate case the thresholds accept — and it also bites on TreeSHAP. |
| **F5** | LIME **violates Eq. (6)'s stated precondition** ("delivered views deterministic conditional on $x$", `paper.tex:594`). Self-stability $B_{\text{self}}=0.590$ with **zero** identical repeats, against $B=0.595$ under a 1 % input perturbation — i.e. essentially all its measured instability is its own sampling noise, not the adversary's. 16× more neighbourhood samples does not fix it (0.603 → 0.592 → 0.616 at 500/2000/8000). | E4b | Any $B$ reported for a stochastic explainer must be labelled as including explainer nondeterminism, or the precondition must be enforced. |
| **F6** | Eq. (10)'s optional $[0,1]$ normalization **destroys** the role ordering it is meant to make comparable. Raw $D$ is monotone in privilege on **0** violations (public 3.847 → designer 0.441); normalised $D$ is non-monotone on **382 / 600** role pairs and is *flat* (0.712 / 0.709 / 0.710 / 0.700), reversing outright for the random control. | E6 | State that the normalization is per-view and carries no cross-role meaning, or drop it. Prop. `lp`'s own assumption (view monotonicity in $k$) holds on **0** violations for every explainer. |
| **F7** | BODMAS separability is a **collection artefact**, and E9 now shows it is specific to BODMAS. Eight of the nine EMBER feature groups reach AUC ≥ 0.986 alone on BODMAS (`exports` is the exception at 0.660); on EMBER-2018, trained at the same size with the same detector, **none of the nine does**, the best being `sections` at 0.931. | E1, **E9** | All results are conditioned on an unusually easy detector, and that ease does not transfer. Must appear as a threat to validity, as `EVIDENCE_GATE.md` already requires. The threshold is 0.986, not 0.987: `byte_entropy` lands at 0.98699 on BODMAS, so at 0.987 the count is seven, not eight. |
| **F8** | The release game is solvable once its payoffs are filled with measured quantities, and its equilibrium is driven by one term: release width is nondecreasing in the evidence weight $\eta$ and nonincreasing in the disclosure weight $\rho$, and at $\eta=0$ it sits at the lattice minimum ($k=5$) for every $\rho$ tested. The SSE is **pure** at all 18 settings (randomisation gain $\le4\times10^{-16}$), and the adversary plays pure observation ($M_4$) at all 15 defender rows. | E8 | Answers M-8: report a computed equilibrium for one finite instance, and keep the non-constructive boundary explicit. An evidence obligation is the only thing that buys release. |

---

## 2. Claim-to-evidence matrix

Each row states what was measured, and — per SEOS — the strongest wording the evidence supports.

| Claim | Purpose served | Measured | Status | Admissible wording |
|---|---|---|---|---|
| **C1** contract slots take real values | instantiability | 5 of 7 slots filled on a real detector: detector, method, audience (4 roles), adversary ($M_1$), disclosure. $\Omega$ (evidence obligation) and $A_\Gamma$/$\Psi$ (actionability) have no BODMAS referent. | **Partial, as gated** | "instantiated on a static-malware detector; the provenance and actionability slots are specified but not exercised" |
| **C2** risk functional + admissibility non-vacuous | non-vacuity | $F$, $B$, $D$ computed on 500 alerts × 4 explainers; admissibility surface over $11^3$ threshold triples × 4 roles = 5,324 cells per explainer. Neither trivially satisfied (0 % at defaults) nor trivially violated (100 % at $\tau_d\ge0.9$). | **Supported, with F3** | "the conditions are satisfiable and non-trivial; the numeric thresholds alone do not exclude a degenerate explanation, which is what Eq. (7) is for" |
| **C2** Eq. (7) calibration is operative | non-vacuity | 200 alerts × 4 explainers × 4 values of $k$, both readings of $\Phi$. Violated by every explainer at every $k$ under the strict reading; TreeSHAP's total failures fall to 1/200 at $k=50$. | **Supported (F4)** | "the calibration condition rejects instantiations that the thresholds admit, and is not satisfied outright by any explainer measured here" |
| **C3** threat class $M_1$ | non-vacuity | 3 perturbation families × 4 budgets × 20 perturbations, 500 alerts. $B$ falls monotonically with budget (0.980 → 0.901 append-bytes) and with $\lvert\Delta_z\rvert$ (0.957 → 0.947). | **Supported for $M_1$ only** | "exercises the input-only adversary; the observing adversary $M_4$ is exercised in E8; $M_2$ and $M_3$ are not exercisable on BODMAS" |
| **C4** release game, existence of SSE | tractability on a case | E8: a $15\times13$ instance with four payoff terms measured and five stipulated; SSE solved by two independent methods agreeing to $4\times10^{-16}$, over 18 $(\eta,\rho)$ settings. | **Supported for one finite instance (F8)** | "an equilibrium is computed for a finite instance built from measured quantities; the proposition remains non-constructive in general, and the instance is solved by enumeration" |
| **C5** Theorem 1 non-vacuous, slack quantified | tightness | 138 configurations, 0 violations, median ratio 0.029 (0.064 under the Dobrushin constant). | **Supported (F1, F2)** | "the bound holds on every measured configuration and is loose by a median factor of ~34 (~16 under a kernel-specific constant)" |
| **C5** Prop. `lp` dominance | instantiability | Raw $D$: 0 violations, all explainers. Normalised $D$: 382–600 violations. View monotonicity in $k$: 0 violations. | **Supported for the raw witness (F6)** | "the dominance holds for the disclosure witness as defined in Eq. (10); the optional normalization is not order-preserving across roles" |
| Props `mono`, `suff`, `local`, `cover`, `marginal` | — | Definitional or negative results; nothing to measure. | **Proved, not tested** | say so plainly |
| Prop. `sse` | tractability on a case | See C4 / E8. | **Proved; instantiated once** | say so plainly |

---

## 3. Results by experiment

### E1 · Substrate and separability — `results/E1_substrate.json`

Temporal split at **2020-07-01**, train 104,578 / test 26,145 (3,712 rows with unparsable
timestamps dropped, not dumped into test — pinned by `tests/test_data_split.py`). HistGB, seed 42:
**AUC 0.9999439**, fit 106 s, alert rate 0.5075, and **every alert is true malware**
(`alert_malware_fraction = 1.0`). 1,000 alerts explained.

Single-group AUC (train on that group alone):

| group | AUC | group | AUC |
|---|---|---|---|
| data_directories | 0.9976 | strings | 0.9923 |
| header | 0.9955 | general_info | 0.9922 |
| sections | 0.9951 | byte_histogram | 0.9906 |
| imports | 0.9942 | byte_entropy | 0.9870 |
| | | **exports** | **0.6599** |

Eight of nine groups separate the corpus almost perfectly on their own. A temporal split cannot
remove that, so it is a property of how BODMAS was collected (**F7**). Seed spread over 5 seeds:
AUC 0.999949 ± 0.000003.

### E2 · Faithfulness $F_\Gamma$ — `results/E2_faithfulness.json`

Deletion against a bank of 5 **verified-benign donors** (labelled benign *and* scored benign,
max score 9.4e-4). $n=200$, the common subset across all four explainers.

| explainer | $F$ mean ± sd | median flip-$k$ | mean conf. at $k=20$ |
|---|---|---|---|
| treeshap | **0.99942 ± 0.00018** | **7.5** | 0.221 |
| lime | 0.84125 ± 0.07319 | 200 | 0.680 |
| constant | 0.71362 ± 0.05999 | 500 | 0.995 |
| random | 0.58342 ± 0.12280 | 1000 | 0.994 |

$F$ saturates for TreeSHAP (sd 1.8e-4), so the deletion curve — not $F$ — is the discriminating
statistic: deleting the top **7.5** features flips the median TreeSHAP alert, against 1000 for
random. Baseline confidence 0.9949; flip rate 1.0 for all four, i.e. *some* deletion always flips
the decision, which is why the flip *budget* is the informative number.

Note the constant control scores $F=0.714$, **above** random: a fixed global ranking is partially
faithful on a detector this separable. $F$ alone is therefore weak evidence of a *local*
explanation.

### E3 · Eq. (7) calibration — `results/E3_calibration.json`, `results/E3_calibration_strict.json`

$\Phi$ runs opposite to $F$ (keep top-$k$ over a benign donor, discard the rest), so the two can
disagree; $\Phi\in[0,1]$ is the fraction of donors under which the top-$k$ alone reproduces the
decision, i.e. **robust** sufficiency. $\tau_f=0.5$, $n=200$.

Eq. (7) asks for $\Phi=1$. With a bank of donors rather than a single one, "violation" admits two
readings, and reporting only one of them misstates the count by up to 62×, so both are recorded:

- **strict** — $F\ge\tau_f$ and $\Phi<1$: the top-$k$ fails under at least one donor;
- **total failure** — $F\ge\tau_f$ and $\Phi=0$: it fails under every donor.

| explainer | $k$ | sufficiency $\overline\Phi$ | $F$-pass | strict violations | total failures | $\Phi=1$ rate |
|---|---|---|---|---|---|---|
| treeshap | 5 | 0.186 | 1.000 | 200 | 59 | 0.000 |
| treeshap | 10 | 0.194 | 1.000 | 200 | 80 | 0.000 |
| treeshap | 20 | 0.492 | 1.000 | 161 | 12 | 0.195 |
| treeshap | 50 | **0.913** | 1.000 | **62** | **1** | **0.690** |
| lime | 50 | 0.188 | 1.000 | 200 | 66 | 0.000 |
| random | 50 | 0.001 | 0.780 | 156 | 155 | 0.000 |
| **constant** | any | **0.000** | **1.000** | **200** | **200** | **0.000** |

Reverse violations (sufficient but $F$ fails): **0** everywhere, for every explainer and $k$.

Three readings, all worth stating. (i) The condition is **not** a formality — under the strict
reading **no** explainer satisfies it on all alerts at any $k$ tested, so an instantiation must
earn it rather than assume it. (ii) TreeSHAP's total failures do fall to 1/200 at $k=50$, so the
condition is *approachable* by enlarging the view — which is itself a disclosure cost, and links
Eq. (7) directly to $D_\Gamma$. (iii) It is the **only** part of the framework that rejects the
constant control (**F4**), which passes $F$ and $B$ outright.

`E3_calibration_strict.json` reproduces run 2's substrate exactly — its total-failure counts match
`E3_calibration.json` term for term (59 / 80 / 12 / 1 for TreeSHAP) — and adds the strict counts
and the $\Phi$ distribution.

### E4 · Robustness $B_{\Gamma,r}$ under $M_1$ — `results/block_*.json`

3 families × 4 budgets × 20 perturbations, 500 alerts (TreeSHAP and both controls); LIME reduced
to 1 family / 1 budget / 5 perturbations / 100 alerts and labelled as such in
`coverage_notes` — LIME costs ~0.4 s per instance against TreeSHAP's ~0.001 s.

$B$ mean (TreeSHAP):

| family | 0.001 | 0.005 | 0.01 | 0.05 |
|---|---|---|---|---|
| append_bytes | 0.9795 | 0.9592 | 0.9465 | 0.9007 |
| add_imports | 0.9901 | 0.9712 | 0.9558 | 0.9060 |
| generic (control) | 0.9841 | 0.9571 | 0.9363 | **0.8543** |

$B$ falls monotonically in the budget, and the two structured families sit **above** the generic
control at 0.05 (0.901 / 0.906 vs 0.854) — functionality-preserving PE edits perturb the
explanation *less* than an unconstrained edit of the same size. Because $B$ is a supremum it can
only fall as $\lvert\Delta_z\rvert$ grows, and it does: 0.9566 → 0.9465 from 1 to 20 perturbations
(append_bytes). **A published $B$ without its perturbation-set size is not interpretable.**

Controls: constant **1.0000 ± 0.0000** at every family and budget; random **0.575** at every family
and budget — flat, because its instability is self-generated and the perturbation adds nothing.

### E4b · Self-stability — `results/E4b_self_stability.json` *(added after run 2)*

$B$ under an **identity** perturbation set: same rows, $K$ repeat explanations. Eq. (6) is stated
for views deterministic conditional on $x$ (`paper.tex:594`); this measures whether that holds.

| explainer | bitwise-identical repeats | $B_{\text{self}}$ | $B$ under 1 % perturbation |
|---|---|---|---|
| treeshap | 5/5 | **1.0000** | 0.9465 |
| constant | 5/5 | **1.0000** | 1.0000 |
| random | 0/5 | 0.5797 | 0.5754 |
| **lime** | **0/5** | **0.5903** | 0.5951 |

LIME neighbourhood sweep, $B_{\text{self}}$: 500 → 0.6032, 2,000 → 0.5921, 8,000 → 0.6161. Sixteen
times the sampling budget does not restore determinism, so this is not an under-sampling artefact
at any budget affordable here (**F5**). For LIME, $B_{\text{self}}=0.590$ and $B$ under a 1 %
perturbation is $0.595$ — the perturbed measurement is marginally *higher*, i.e. the input
perturbation contributes nothing distinguishable from noise, and the entire 0.41 gap from 1 is the
explainer explaining the same input differently each time.

### E5 · Theorem 1 bridge — `results/block_*.json` → `E5_bridge`

$L_\pi=1$ **exactly**, by the $\ell_1$ non-expansiveness of a column-stochastic kernel — proved in
`src/txai_exp/release.py` and verified numerically on 200 random pairs in `tests/test_metrics.py`, so
the measured slack is not an artefact of estimating a constant.

**138 configurations, 0 violations** (4 explainers × families × budgets × 3 kernel temperatures,
plus 2 detector ablations and 5 seeds).

| explainer | configs | median LHS/RHS | max | mean RHS |
|---|---|---|---|---|
| treeshap | 36 | **0.0294** | 0.2427 | 0.0549 |
| lime | 3 | 0.0028 | 0.0065 | 0.4049 |
| random | 36 | 0.0009 | 0.0034 | 0.4248 |
| constant | 36 | 0/0 (both sides exactly 0) | — | 0 |

Tightness *falls* as the explanation gets worse: the bound is loosest exactly where a guarantee
would matter most. Mean LHS by kernel temperature (TreeSHAP): 0.00660 / 0.00297 / 0.00086 at
$T=0.25/1/4$ — while RHS is **identical** across all three, because $L_\pi=1$ cannot see the
kernel. That observation motivates E5b.

### E5b · Where the slack comes from — `results/E5b_kernel_constant.json` *(added after run 2)*

The tight TV→TV constant for a column-stochastic map is its Dobrushin coefficient
$\delta(W)=\max_{j,k}\tfrac12\lVert W_{\cdot j}-W_{\cdot k}\rVert_1\le1$, and the manuscript's
kernel is $\pi(\cdot\mid z)=(GA)p$ with $A$ the group-aggregation matrix, so
$\delta(GA)=\delta(G)$. The Theorem 1 derivation goes through verbatim with $\delta$ in place of
$L_\pi$.

| kernel | $L_\pi$ (manuscript) | $\delta$ | empirical max ratio over 2,000 random pairs | violations |
|---|---|---|---|---|
| $T=0.25$ | 1 | 0.9823 | 0.9332 | 0 |
| $T=1.0$ | 1 | 0.6055 | 0.5033 | 0 |
| $T=4.0$ | 1 | 0.1928 | 0.1558 | 0 |

Substituting $\delta$ into the recorded rows: TreeSHAP median tightness **0.0294 → 0.0643**
(factor 2.19), LIME 0.0028 → 0.0046, random 0.0009 → 0.0015, **0 violations** anywhere. So about
half the slack is recoverable by a constant the paper can compute in closed form; the remaining
~16× is genuine contraction in the release kernel and the world-averaging inside $\mathrm{Adv}^{TV}$.

### E6 · Disclosure $D_\Gamma$ and least-privilege dominance — `results/E6_disclosure.json`

Eq. (10) as written, $D_\Gamma(z,R)=\sum_{c\in\mathrm{comp}(z)}w_R(c)$, top-$k$ = 20, 200 alerts.

| explainer | raw monotone | raw violations | mean raw $D$ public → designer | normalised monotone | normalised violations |
|---|---|---|---|---|---|
| treeshap | **yes** | 0 | 3.847 → 0.441 | **no** | 382 / 600 |
| lime | yes | 0 | 4.057 → 0.475 | no | 376 |
| random | yes | 0 | 3.745 → 0.450 | no | 512 |
| constant | yes | 0 | 3.200 → 0.400 | no | 600 |

Normalised $D$ is flat across roles (TreeSHAP 0.7124 / 0.7087 / 0.7105 / 0.7000) and **reverses**
for the random control (0.694 → 0.714). The normalization divides by a role-dependent total, so it
compares a view against that role's own maximum, not against other roles (**F6**). Prop. `lp`'s
assumption — $D$ monotone in the view's own size — holds on **0** violations for every explainer
(TreeSHAP 0.318 / 0.574 / 0.709 / 0.823 at $k=5/10/20/50$).

### E7 · Admissibility (Def. 3, three of four conditions) — `results/block_*.json` → `E7_admissibility`

$A_\Gamma$ is **excluded and marked assumed**: BODMAS supports no response playbook, and inventing
one would fabricate the governance layer the paper contributes. 5,324 cells per explainer
($11^3$ thresholds × 4 roles).

At the manuscript's defaults $(\tau_f,\tau_b,\tau_d)=(0.5,0.8,0.5)$ the rate is **0.000 for every
explainer and every role** — the binding constraint is $D$, whose normalised value is ≈0.6–0.76,
above $\tau_d=0.5$. Loosening only $\tau_d$ to 0.8 (analyst role):

| $\tau_d$ | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|
| treeshap | 0.000 | 0.136 | 0.300 | 0.944 | 1.000 |
| **constant** | 0.000 | 0.000 | **1.000** | **1.000** | 1.000 |
| lime | 0.030 | 0.030 | 0.030 | 0.030 | 0.030 |
| random | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

The constant control reaches 100 % admissibility at a *stricter* $\tau_d$ than TreeSHAP, and is
admissible over 28.1 % of the whole surface against TreeSHAP's 22.5 % (**F3**). $\tau_b=0.8$ does
exclude the random control everywhere, so the robustness threshold works against noise — just not
against a constant.

Detector ablation at defaults: xgboost 0.028 for all roles, randomforest 0.028 (0.042 for
designer). Identical across roles, which is the flat-normalised-$D$ artefact of F6 again.

### E8 · Release-game equilibrium — `results/E8_release_game.json`, `results/E8_decoupled_control.json` *(added 2026-08-03)*

Answers round-2 comment M-8, which asked for a computed equilibrium or a downgrade of the release
game to a formulation. A $15\times13$ finite instance: defender = 5 release widths
$\{5,10,20,50,100\}$ × 3 kernel temperatures; adversary = pure observation ($M_4$) + 12 $M_1$
transformations (3 families × 4 budgets). $r$ is one role-indexed mechanism, so the untrusted role
gets $\lfloor k/4 \rfloor$ of the analyst width.

**Measured per cell:** $q^{i,a}_\sigma$ (joint world–action law), $D_\Gamma$ of the public view
(Eq. 10), attacker-visible content (fraction of the 9 EMBER groups revealed), and evidence-obligation
failure (donor sufficiency $\Phi$ on the delivered analyst view).
**Stipulated and swept:** $\ell$, $\kappa$, $\beta$, $\eta$, $\rho$. Declared in
`src/txai_exp/game.py::GameInstance`; $\eta,\rho$ swept over $3\times6$ settings.

| finding | number |
|---|---|
| SSE at $\eta=\rho=1$ | pure: $k=50$, softmax $t{=}0.25$; adversary `M4_observe`; $U_D=-1.1384$ |
| MILP vs multiple-LP | agree to $4.4\times10^{-16}$ at all 19 solves (DOBSS \[Paruchuri 2008] vs Conitzer–Sandholm 2006; no shared code path) |
| randomisation gain | $\le 4.4\times10^{-16}$ everywhere — the SSE is **pure** on this instance |
| release width vs $\eta$ | nondecreasing; at $\eta=0$ it is $k=5$ at every $\rho$ |
| release width vs $\rho$ | nonincreasing; $\eta{=}1$: $100,100,100,50,5,5$ over $\rho=0\ldots4$ |
| what pays for release | evidence failure $0.861\to0.041$ as $k:5\to100$; $D_\Gamma$ $0.116\to0.730$; decision loss $0.284\to0.394$ |
| adversary's best response | `M4_observe` at **all 15** defender rows; transformations move $q_{\text{target}}$ by $\le0.019$; break-even content price for the strongest $M_1$ is $\beta=1.069$ vs $1.000$ |
| world mass | benign 0.000 / malware 1.000 → $\ell$ collapses to one column (a consequence of F7, not of the game) |
| wall clock | 364 s, seed 42, CPU |

**Decoupled control** (`E8_decoupled_control.json`). The first instance held the analyst's own view
fixed at top-20 and varied only the public width. Release is then pure cost and the equilibrium is
"disclose nothing" ($k_A=0$) at all 18 settings, $U_D=-0.9306$. Retained rather than deleted: it is
the measurement showing that the cross-role coupling, not the payoff shape, is what makes the game
non-trivial.

**Boundaries.** Solved by enumeration, so it demonstrates tractability on a case and leaves
Prop. 6 non-constructive in general. $\Omega$ is instantiated as one computable sufficiency test and
$\Delta_m$ as one seeded transformation per budget — a restriction of the game, and a wider
$\Delta_m$ can only raise the adversary's value. The $M_4$-dominates-$M_1$ ordering depends on
stipulated $\beta,\kappa$ as well as on the measurement.

Tests: `tests/test_game.py`, 14 cases, incl. a constructed game where mixing strictly beats every
pure commitment (so the zero randomisation gain above is a property of the instance, not of the
solver) and MILP–LP agreement on 12 random bimatrices.

---

### E9 · Does F7 survive a second corpus? — `results/E9_ember2018.json` *(added 2026-08-03)*

EMBER-2018, feature version 2, vectorised from the released JSONL to the matrix directory
(`$TXAI_DATA_DIR/ember2018_matrix`; 1,000,000 × 2,381; train 800,000 / test 200,000; 400,000 benign /
400,000 malware / 200,000 unlabeled; train ≤ 2018-10, test 2018-11…12 — the released split is
already temporal, so it is used as is). Its authors state the corpus was assembled so the
train/test sets "would be harder for machine learning algorithms to classify", which is exactly
the substrate F7 needs to be tested against.

**The comparison holds one variable.** Same detector (`histgb`, `max_iter=200`, `lr=0.1`,
seed 42), same per-group setting (`max_iter=40`), and the *same training-set size* — 104,578
rows, read from `E1_substrate.json` rather than typed, so it cannot drift if E1 is re-run. Using
all 600,000 labeled EMBER rows would have trained the replication on six times E1's data and made
a per-group difference unattributable between corpus and sample size.

| feature group | BODMAS (E1) | EMBER-2018 (E9) | Δ |
|---|---|---|---|
| data_directories | 0.9976 | 0.9175 | −0.0801 |
| header | 0.9955 | 0.8856 | −0.1099 |
| sections | 0.9951 | **0.9305** | −0.0645 |
| imports | 0.9942 | 0.9020 | −0.0922 |
| strings | 0.9923 | 0.9256 | −0.0667 |
| general_info | 0.9922 | 0.9098 | −0.0824 |
| byte_histogram | 0.9906 | 0.9292 | −0.0614 |
| byte_entropy | 0.9870 | 0.9177 | −0.0693 |
| exports | 0.6599 | 0.5578 | −0.1021 |
| **groups ≥ 0.986 alone** | **8 of 9** | **0 of 9** | |
| **full detector** | 0.999944 | 0.988315 | −0.0116 |

Every group falls, by 0.06 to 0.11, and the count clearing the threshold goes from eight to
**zero**. The best single group on EMBER-2018 (`sections`, 0.931) is below the *weakest*
qualifying group on BODMAS (`byte_entropy`, 0.987). `exports` is the weakest on both corpora, so
what transfers is the *ordering* of the groups, not the level.

**The subsample is a quantified caveat, not an asserted one.** Stage 3 trained the full detector
at four sizes on EMBER-2018:

| train rows | 25,000 | 50,000 | 100,000 | 104,578 |
|---|---|---|---|---|
| full-detector AUC | 0.983860 | 0.986492 | 0.987659 | 0.988315 |

Quadrupling the training data moves AUC by 0.0045, and the last 4,578 rows buy 0.0007. The curve
is flat at the size used, so the gap to BODMAS is not a training-size effect: closing a 0.06–0.11
per-group deficit at that rate is not plausible on any amount of data this corpus contains.

**Reading.** F7's second clause — that near-perfect *single-group* separability is a property of
how the corpus was collected — is confirmed and now **narrows to BODMAS**. It is not a property
of static PE features in general. The manuscript's threat-to-validity is therefore
strengthened, not weakened: the substrate really is unusually easy, and there is now a measured
comparison saying so rather than an argument.

What E9 does **not** establish: no explanation, faithfulness, calibration, robustness, or
release-game quantity was recomputed on EMBER-2018. E9 replicates E1 only. Whether the
explanation-layer findings hold on a harder substrate is untested and must not be implied.

Cost and provenance: vectorisation 2.0 min (202 chunks, 7 workers, 0 failed, 0 unparsable);
E9 itself one attempt, no retry, fit 207 s, peak RSS 2.82 GB. Feature extractor vendored
byte-identical from upstream EMBER (`vendor/PROVENANCE.md`, sha256 recorded) with legacy
`FeatureHasher` semantics restored in `txai_exp/ember/compat.py` so the vectors land in the same
feature space as BODMAS — settled by measurement against BODMAS's entry-name block, not by
preference. Tests: `tests/test_ember.py`, 17 cases.

> A first attempt ran at 300,000 training rows and was stopped after 73 minutes without
> producing an AUC. Its worker threads were at ~96 % system time (≈12 min kernel vs 27 s user
> each): a 16 GB machine already 13 GB into swap, not slow arithmetic. Marginal cost is
> ~17.5 KB per training row. Matching E1's size fixed it and is the better experiment anyway.
> A hypothesis that `HistGradientBoostingClassifier` duplicates float32 input as float64 was
> tested and **rejected** — measured one process per dtype at 40k × 2,381, float32 peaks at
> 1.66 GB and float64 at 2.26 GB, so this sklearn bins straight from float32.

---

### Ablations

| ablation | result |
|---|---|
| **A1** explainer | TreeSHAP / LIME / random / constant throughout; see every table above |
| **A2** detector | xgboost AUC 0.999913, $B$ 0.962 (append_bytes @0.01), tightness median 0.027; randomforest AUC 0.999665, $B$ 0.910, tightness 0.0102. Looseness is not a detector artefact. |
| **A3** perturbation family | append_bytes / add_imports / generic — see E4 |
| **A4** thresholds | full $11^3$ surface, E7 |
| **A5** perturbation budget | 0.001 / 0.005 / 0.01 / 0.05, E4 |
| **A6** $\lvert\Delta_z\rvert$ | 1 / 5 / 10 / 20, E4 |
| **A7** seed (×5) | AUC 0.999949 ± 0.000003; tightness median 0.0811 ± 0.0176; $B$ 0.9511 ± 0.0025. No headline number is seed-fragile; tightness is the most variable (±22 % relative). |
| **A8** LIME neighbourhood | 500 / 2,000 / 8,000 samples, E4b |
| **A9** kernel temperature | 0.25 / 1.0 / 4.0, E5 and E5b |

---

## 4. Correction record

Two defects were found **after** the first full run (2026-08-02, 27.7 min) and fixed before run 2.
Run 1 is retained at `results_run1_superseded/` with its own README, because a discarded run that
is not reported is a silent selection of results.

1. **`pipeline.explain` rebuilt the explainer on every call.** The seeded random control therefore
   replayed identical draws, its perturbed attributions came back bit-identical to its clean ones,
   and it was recorded with $B=1.000,\ \mathrm{sd}=0$ at every family and budget — an artefact of
   object construction, not a measurement. LIME was affected in the opposite direction: its own
   sampling noise was reset per call, inflating its $B$ from **0.595** to **0.955**. Fixed by
   caching the explainer instance (which also stopped re-parsing the SHAP tree 240× per block, and
   cut TreeSHAP batches from ~2 s to ~0.5 s). Pinned by
   `tests/test_explainers.py::test_pipeline_explain_does_not_reset_a_stateful_explainer`.
2. **`environment()` probed `module.__version__`.** `lime` 0.2.0.1 defines none, so run 1's
   reproducibility record says `"lime": "absent"` in the same file as a 345 s LIME block. Fixed to
   read distribution metadata.

Both defects inflate a *robustness* number, which is the direction that would have flattered the
framework. The `constant` control was added in the same change: without an explainer that is
input-independent by construction, F3 and F4 are invisible.

Found while adding E9 (2026-08-03), in the reporting rather than in any measurement:

3. **F7 was stated at the wrong threshold in this file.** The headline row read "AUC ≥ 0.987",
   which does not match its own evidence: `byte_entropy` is 0.98699 on BODMAS, so at 0.987 the
   count is seven of nine, not eight. `paper.tex` (sec:numlimits) had it right at 0.986 all
   along. Corrected here, and `run_ember_transfer.py` now holds the threshold in one constant applied
   to both corpora, with the BODMAS count recomputed from `E1_substrate.json` instead of typed —
   a hardcoded "8 of 9" would keep reading as true after E1 changed.
4. **`\pm` denotes a different estimator in different artifacts.** A7 above reports
   $B = 0.9511 \pm 0.0025$ (population sd, `ddof=0`, what `metrics.summarise` stores);
   `paper.tex` line 2102 reports $0.9511 \pm 0.0027$ (sample sd, `ddof=1`) for the same
   quantity. Neither states which. Two further manuscript defects share the cause: "the
   detector's AUC varies by $4\times10^{-6}$" is a standard deviation described as a range (the
   range is $9.0\times10^{-6}$), and LIME and the constant control are reported without spread
   while TreeSHAP and the random control get one — the two omitted are the widest, at roughly
   400× and 340× TreeSHAP's. Open manuscript work; recorded in
   `results_interpretation_20260802/RESULTS_ANALYSIS_QUALITY_REPORT.md`.

Earlier corrections, made during construction and already recorded in the code:

- The deletion reference was initially a coordinate-wise **median**, which scores 0.99999 (malware)
  and made $F\approx10^{-5}$ for every explainer, with random *above* TreeSHAP. Replaced by a bank
  of verified-benign donors. Rejected alternatives and their measured scores are in
  `metrics.faithfulness_F`'s docstring.
- A test asserted that $D$ falls as privilege rises. It failed (47/41/36 violations). Reading
  Eq. (10) and Prop. `lp` in the source showed the manuscript never claims that, and that the
  per-role normalization makes it false — which became F6. The implementation, the function name
  (`role_disclosure_profile`), and the test were all corrected rather than the failure patched.

---

## 5. Threats to validity

1. **One dataset, and an unusually easy one (F7) — now measured, not argued.** Eight of nine
   feature groups separate BODMAS alone at AUC ≥ 0.986. E9 repeated that measurement on
   EMBER-2018 at the same training size with the same detector and **none of the nine clears the
   threshold**, the best reaching 0.931. Every number in this file therefore describes
   explanations of a detector that is near-perfectly separable *on this corpus specifically*.
   No claim may generalise past a single static-PE benchmark — and E9 makes that a finding
   rather than a caution. What E9 does not do is re-measure the explanation layer on the harder
   corpus, so nothing above is transferred either.
2. **All alerts are true positives** (`alert_malware_fraction = 1.0`), so nothing here says
   anything about explaining false positives — which is most of an analyst's real workload.
3. **$A_\Gamma$, $\Omega$, the claim-evidence graph, the role lattice with real consequences, and
   threat classes $M_2$–$M_4$ are untested**, by design. The programme exercises the *scoring*
   layer only. Any sentence implying the governance layer was validated is unsupported.
4. **LIME's grid is reduced** (1 family, 1 budget, 5 perturbations, 100–200 alerts) for cost
   reasons, recorded in each block's `coverage_notes`. LIME numbers must not share a column with
   TreeSHAP's without that note.
5. **$\Phi$ is one sufficiency instantiation**, not the only one. Eq. (7)'s violation rate is a
   statement about this $(F,\Phi)$ pair; a different $\Phi$ would give different rates. The
   *existence* of a violating pair is what makes the condition non-vacuous, and that is all F4
   claims.
6. **Human factors are absent.** No claim of analyst trust, usability, or comprehensibility is
   supported by anything here.
7. **Both corpora are third-party.** BODMAS and EMBER-2018 are used as published, read-only
   and unmodified; neither is redistributed here. Any property of the corpora themselves --
   labelling, sampling, family coverage -- is inherited from their publishers, not established
   by this programme.

---

## 6. What to do with this in the manuscript

Ordered by how much of the paper each touches.

1. **Adopt or refuse the Dobrushin constant (F2).** It is a strict improvement to Theorem 1's
   statement, costs one line of proof, and is verified on 138 configurations.
2. **Add the calibration/threshold separation (F3 + F4) as the experimental section's spine.** It
   is the one result that justifies the framework's architecture: the numbers alone accept a
   constant vector, and Eq. (7) is what rejects it.
3. **Fix or drop Eq. (10)'s normalization (F6).** As written the optional normalization is not
   order-preserving across roles, and the manuscript's dominance discussion reads as if it were.
4. **State the bound's looseness (F1) rather than leaving it implied.** Median 34× is a finding.
5. **Add Eq. (6)'s determinism precondition to the discussion of stochastic explainers (F5)**,
   with LIME as the worked case.
6. **Promote F7 from a limitation to a measurement (E9).** The limitation stands, but it no longer
   has to be argued: on EMBER-2018, at the same training size with the same detector, none of the
   nine groups clears the threshold that eight clear on BODMAS. Say that, cite the numbers, and
   keep the title, abstract, and contributions from generalising past one static-PE corpus. The
   honest form of the caveat is now "measured on two corpora, and the easy one is BODMAS" —
   stronger than the hedge it replaces, and cheaper to defend under review.

Per SEOS separation of duties, whoever writes these claims into `paper.tex` must read the numbers
from `results/*.json`, not from this summary's prose.
