"""Emit the pgfplots figure blocks for Section~VII from ``results/*.json``.

Every coordinate written into the manuscript is transcribed here from the JSON
produced by ``run_scoring_layer.py``; no number in the emitted LaTeX is typed by
hand. Re-running this script after a re-run of the experiments regenerates the
figures, so a changed measurement cannot silently leave a stale plot behind.

The four figures are chosen so that each carries information that no table
carries, per the de-duplication constraint on Section~VII:

  fig:deletion    E2 deletion curves      -- retires the flip-$k$ column
  fig:robustness  E4 budget sweep         -- retires the $B_{\\Gamma,r}$ column
  fig:admissible  E7 threshold slice      -- retires the admissible column
  fig:bridge      E5/E5b bound vs measured -- retires Table~\\ref{tab:numbridge}

One table is emitted here for the same reason -- ``tab:numgame`` carries E8's
equilibrium over the swept weights, and its cells are lattice levels rather than
coordinates, which a plot would render less legibly than a grid does.

Usage: python3 make_figures.py > figures_numerical.tex
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

RESULTS = Path(__file__).resolve().parents[1] / "results"

# Okabe-Ito, chosen for colour-vision deficiency safety; every series also
# carries a distinct marker so the figures survive greyscale printing.
#
# The constant control is black rather than Okabe-Ito vermillion: vermillion
# (213,94,0) and the orange (230,159,0) carrying LIME are adjacent hues, and in
# Figs. 3 and 4 the two series run close enough that a reader has to consult the
# markers to tell them apart. Black also reads as what the series is -- the
# degenerate control the argument is built against -- so the reference lines in
# Fig. 3 are drawn in grey to leave it unambiguous.
MAP_STYLE: Dict[str, Tuple[str, str]] = {
    "treeshap": ("oiblue", "*"),
    "lime": ("oiorange", "square*"),
    "random": ("oipurple", "triangle*"),
    "constant": ("black", "diamond*"),
}
MAP_LABEL: Dict[str, str] = {
    "treeshap": "TreeSHAP",
    "lime": "LIME",
    "random": "random",
    "constant": "constant",
}
# The "TreeSHAP, " prefix the legend used to repeat on all three entries is in
# the caption instead: it cost a third of the legend's width to say the same
# thing three times, and the legend has to fit beside the curves.
FAMILY_STYLE: Dict[str, Tuple[str, str, str]] = {
    "append_bytes": ("oiblue", "*", r"\textsf{append\_bytes}"),
    "add_imports": ("oigreen", "square*", r"\textsf{add\_imports}"),
    "generic": ("oisky", "triangle*", r"\textsf{generic}"),
}


#: Set by `--source` to a nested result file written by `run_numerical_instantiation.py`.
#: `load` then serves the stage names out of it instead of reading the flat
#: per-stage files, so one figure set cannot mix two corpora: with a source
#: selected, a stage the source does not carry is an error rather than a
#: silent fallback to whatever the last BODMAS run left on disk.
_SOURCE: Dict[str, Any] = {}
_SOURCE_NAME = ""

#: The sharpening file `fig_bridge` reads; overridden by `--bridge`.
BRIDGE_FILE = "E5b_kernel_constant"


def use_source(name: str) -> None:
    """Serve subsequent `load` calls out of one nested result file."""
    global _SOURCE, _SOURCE_NAME
    path = RESULTS / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"missing result file: {path}")
    with path.open(encoding="utf-8") as handle:
        nested = json.load(handle)
    _SOURCE = dict(nested)
    _SOURCE.update({f"block_{k}": v for k, v in nested["blocks"].items()})
    _SOURCE_NAME = name


def load(name: str) -> Dict[str, Any]:
    """Read one result file, failing loudly if the experiment never ran."""
    if _SOURCE:
        if name in _SOURCE:
            return _SOURCE[name]
        # Not in the nested file: only the sharpening, which is computed
        # afterwards, legitimately lives beside it. Anything else would be a
        # stage from another corpus.
        if not name.startswith("E5b_kernel_constant"):
            raise SystemExit(f"{name!r} is not in {_SOURCE_NAME}.json; "
                             "refusing to fall back to a file from another run")
    path = RESULTS / f"{name}.json"
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise SystemExit(f"missing result file: {path}")


def coords(points: Iterable[Sequence[float]], digits: int = 6) -> str:
    """Format (x, y) pairs as a pgfplots coordinate list."""
    return " ".join(f"({x:.{digits}g},{y:.{digits}g})" for x, y in points)


def _first_crossing(curve: Dict[str, Any]) -> str:
    """The first grid point at which the mean confidence falls below $0.5$.

    Read off the grid rather than interpolated: the curve is a mean over alerts
    at the $k$ values actually measured, and a crossing quoted between two of
    them would be a number no experiment produced.
    """
    for k, conf in zip(curve["k_values"], curve["mean_confidence_by_k"]):
        if conf < 0.5:
            return f"$k={k}$"
    return "no measured $k$"


def fig_deletion() -> str:
    """E2: mean detector confidence against the number of deleted features."""
    by_map = load("E2_faithfulness")["by_explainer"]
    baseline = by_map["treeshap"]["deletion_curve"]["baseline_confidence"]
    n_alerts = by_map["treeshap"]["deletion_curve"].get(
        "n", by_map["treeshap"]["n"])
    crossing = {name: _first_crossing(by_map[name]["deletion_curve"])
                for name in by_map}
    # The caption speaks of "the two controls" in one breath, which is only
    # honest while they cross together; if a re-run separates them, say so
    # rather than quoting one of the two under a plural.
    if crossing["constant"] == crossing["random"]:
        controls_clause = f"neither control before {crossing['constant']}"
    else:
        controls_clause = (f"the constant control at {crossing['constant']} and "
                           f"the random control at {crossing['random']}")
    plots: List[str] = []
    for name in ("treeshap", "lime", "constant", "random"):
        curve = by_map[name]["deletion_curve"]
        colour, mark = MAP_STYLE[name]
        plots.append(
            f"\\addplot[color={colour},mark={mark}] coordinates {{"
            f"{coords(zip(curve['k_values'], curve['mean_confidence_by_k']))}}};\n"
            f"\\addlegendentry{{{MAP_LABEL[name]}}}"
        )
    return rf"""\begin{{figure}}[t]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[txaiaxis,
  xmode=log, log basis x=10,
  xmin=0.85, xmax=2400, ymin=-0.03, ymax=1.06,
  ytick={{0,0.25,0.5,0.75,1}},
  xlabel={{Features deleted in attribution order, $k$}},
  ylabel={{Mean detector confidence}},
  legend style={{at={{(0.02,0.02)}},anchor=south west}}]
\addplot[black!45,dashed,forget plot,line width=0.5pt,mark=none,domain=0.85:2400]
  {{{baseline:.6g}}};
\addplot[black!45,dotted,forget plot,line width=0.5pt,mark=none,domain=0.85:2400]
  {{0.5}};
\node[anchor=east,font=\tiny,text=black!60,inner sep=1.5pt]
  at (axis cs:2300,0.93) {{undeleted mean}};
\node[anchor=west,font=\tiny,text=black!60,inner sep=1.5pt]
  at (axis cs:1.05,0.55) {{decision boundary}};
{chr(10).join(plots)}
\end{{axis}}
\end{{tikzpicture}}
\caption{{Deletion behavior of the four explanation maps on the ${n_alerts}$
alerts of the instantiation of Section~\ref{{sec:numsetup}}: mean confidence
of the \textsf{{histgb}} detector after deleting the $k$ highest-attributed
features. The two gray guides mark the undeleted mean confidence
(${baseline:.4f}$) and the decision boundary. Read the
curves for their \emph{{ordering and shape}}, not for a per-alert guarantee:
TreeSHAP crosses the boundary at {crossing['treeshap']}, LIME at
{crossing['lime']}, and {controls_clause}, by which point almost the entire
feature vector has been deleted and the crossing says nothing about the ordering
that produced it. LIME's curve is
measured on the reduced grid its cost forced (Section~\ref{{sec:numlimits}}) and
is not directly comparable to the others. Single corpus, single detector, single
seed; the ordering is not asserted beyond this instantiation.}}
\label{{fig:deletion}}
\end{{figure}}"""


def fig_robustness() -> str:
    """E4: robustness score against the adversary's perturbation budget.

    Two panels with independent vertical ranges, not one broken axis. The random
    control sits far below everything else, so a single continuous scale spends
    most of its height on the empty band between them and squeezes the three
    TreeSHAP curves -- the only series that moves, and the reason the figure
    exists -- into a sliver. The earlier draft drew that as one axis with break
    marks, which invites the reader to compare vertical distances across the
    break; nothing in the figure supports such a comparison, and the round-3
    review objected. Separate panels make the two ranges separate claims.
    """
    treeshap = load("block_treeshap")["E4_robustness"]["by_family_and_budget"]
    # The sampling budget goes in the caption rather than being typed there: it
    # is what makes every plotted value an upper estimate of the supremum in
    # (5), so a caption that named the wrong one would misstate the direction of
    # the error.
    cfg = _SOURCE.get("config", {}) if _SOURCE else {}
    n_perturb = cfg.get("n_perturb", 20)
    n_robust = cfg.get("n_robust", 500)

    def control(name: str) -> List[Tuple[float, float]]:
        cells = load(f"block_{name}")["E4_robustness"]["by_family_and_budget"]["append_bytes"]
        return [(float(b), cells[b]["B_mean"]) for b in sorted(cells, key=float)]

    upper: List[str] = []
    for family, (colour, mark, label) in FAMILY_STYLE.items():
        cells = treeshap[family]
        pts = [(float(b), cells[b]["B_mean"]) for b in sorted(cells, key=float)]
        upper.append(
            f"\\addplot[color={colour},mark={mark}] coordinates {{{coords(pts)}}};\n"
            f"\\addlegendentry{{{label}}}"
        )
    colour, mark = MAP_STYLE["constant"]
    upper.append(
        f"\\addplot[color={colour},mark={mark},dashed] coordinates "
        f"{{{coords(control('constant'))}}};\n"
        f"\\addlegendentry{{constant control}}"
    )
    lower_pts = control("random")
    colour, mark = MAP_STYLE["random"]
    lower = (f"\\addplot[color={colour},mark={mark},dashed] coordinates "
             f"{{{coords(lower_pts)}}};")

    # The lower panel gets a range wide enough to show whether the control moves
    # at all. A range fitted tightly to a nearly flat series magnifies sampling
    # noise into an apparent trend, which is the mirror image of the mistake the
    # broken axis made; 0.02 is the smallest span at which the three TreeSHAP
    # movements in the upper panel are still visible, so the two panels resolve
    # motion at comparable scales even though their absolute ranges differ.
    lo = min(y for _, y in lower_pts)
    hi = max(y for _, y in lower_pts)
    mid = 0.5 * (lo + hi)
    span = max(hi - lo, 0.02)
    ticks = [mid - 0.5 * span, mid, mid + 0.5 * span]
    ytick = ",".join(f"{t:.4f}" for t in ticks)
    yticklabels = ",".join(f"${t:.3f}$" for t in ticks)

    return rf"""\begin{{figure}}[t]
\centering
\begin{{tikzpicture}}
\begin{{groupplot}}[txaiaxis,
  % Two panels, not one axis with a break: the ranges are disjoint and nothing
  % here licenses reading a vertical distance across them. ``vertical sep`` is
  % wide enough that the pair does not read as a single interrupted scale.
  group style={{group size=1 by 2, vertical sep=17pt,
                x descriptions at=edge bottom}},
  % ``scale only axis'' so that the two heights below size the plotting boxes
  % themselves: without it pgfplots subtracts the shared x labels from the short
  % lower panel and aborts with a negative plot height. The width then excludes
  % the y label and tick labels, hence 0.82 rather than 0.97:
  % at 0.84 the picture runs 2.4pt past the column.
  scale only axis, width=0.82\columnwidth,
  xmode=log, log basis x=10,
  xtick={{0.001,0.005,0.01,0.05}},
  xticklabels={{$0.1\%$,$0.5\%$,$1\%$,$5\%$}},
  x tick label style={{/pgf/number format/assume math mode=true}},
  y tick label style={{/pgf/number format/assume math mode=true}},
  xmin=0.0008, xmax=0.065]
\nextgroupplot[height=0.40\columnwidth, ymin=0.835, ymax=1.025,
  ytick={{0.85,0.9,0.95,1}},
  ylabel={{$B_{{\Gamma,r}}$: maps under audit}},
  legend style={{at={{(0.03,0.05)}},anchor=south west}}]
{chr(10).join(upper)}
\nextgroupplot[height=0.16\columnwidth,
  ymin={mid - 0.75 * span:.4f}, ymax={mid + 0.75 * span:.4f},
  ytick={{{ytick}}}, yticklabels={{{yticklabels}}},
  ylabel={{$B_{{\Gamma,r}}$: random}},
  xlabel={{Perturbation budget $r$ (fraction of the input)}}]
{lower}
\end{{groupplot}}
\end{{tikzpicture}}
\caption{{Robustness score \eqref{{eq:robustness}} against the adversary's budget,
each point a mean over ${n_robust}$ alerts $\times$ ${n_perturb}$ sampled transformations. Both
panels plot the same quantity on the same horizontal axis, on \emph{{separate}}
vertical ranges: the random control lies far below every other map, and on one
continuous scale it flattens the three curves that actually move into the top
seventh of the panel. The panels are drawn apart rather than as one axis with a
break because no comparison of vertical distances across them is intended or
supported; read each panel's movement against its own scale, and the two
absolute levels from the tick labels. \emph{{Upper}}: TreeSHAP under the three
admitted transformation families (solid) and the constant control (dashed), the
controls measured under \textsf{{append\_bytes}}. \emph{{Lower}}: the random
control, on a range ${1.5 * span:.3g}$ wide, so a flat line here is flat and not a range
fitted to noise. Section~\ref{{sec:numrobust}} reads the three movements and says
what the flat lines do and do not license; every value is computed from a maximum
over ${n_perturb}$ sampled members of $\Delta_z$ rather than the supremum over all of it,
and so over-estimates $B_{{\Gamma,r}}$. LIME
is absent: its cost admitted only the single $1\%$ point, reported in the text.
Single corpus, single detector, single seed.}}
\label{{fig:robustness}}
\end{{figure}}"""


def fig_admissible() -> str:
    """E7: admissible fraction along $\\tau_d$ at the paper's other defaults."""
    plots: List[str] = []
    series: Dict[str, List[Tuple[float, float]]] = {}
    for name in ("constant", "treeshap", "lime", "random"):
        surface = load(f"block_{name}")["E7_admissibility"]["surface"]
        rows = sorted(
            (r for r in surface
             if r["role"] == "analyst" and r["tau_f"] == 0.5 and r["tau_b"] == 0.8),
            key=lambda r: r["tau_d"],
        )
        if len(rows) != 11:
            raise SystemExit(f"{name}: expected 11 tau_d points, got {len(rows)}")
        series[name] = [(r["tau_d"], r["rate"]) for r in rows]
        colour, mark = MAP_STYLE[name]
        plots.append(
            f"\\addplot[color={colour},mark={mark}] coordinates "
            f"{{{coords(series[name])}}};\n"
            f"\\addlegendentry{{{MAP_LABEL[name]}}}"
        )
    # The one comparison the figure exists to make, annotated in-plot rather than
    # left for the caption: the threshold at which the degenerate control is
    # accepted on the largest excess of alerts over the Shapley attribution, and
    # what each reads there. Both come from the data, so a re-run cannot leave a
    # stale annotation behind.
    #
    # The earlier draft looked instead for the first threshold at which the
    # control is admissible on *every* alert. That threshold exists on BODMAS and
    # does not exist on EMBER-2018, where the control's own faithfulness score
    # fails at a third of the alerts, so the search raised StopIteration. The
    # quantity the argument needs is the excess, which is defined either way.
    const, shap = dict(series["constant"]), dict(series["treeshap"])
    taus = sorted(const)
    tau_star = max(taus, key=lambda t: const[t] - shap[t])
    const_at, shap_at = const[tau_star], shap[tau_star]
    excess = const_at - shap_at
    # Where the ordering reverses, if it does. On a corpus where the control
    # dominates throughout there is no such threshold, and the caption must not
    # invent one.
    reversal = next((t for t in taus if t > tau_star and shap[t] > const[t]), None)
    if excess > 0:
        reading = (
            rf"the marked span is the whole of it. At $\tau_d={tau_star:g}$ the "
            rf"constant control is admissible on ${const_at:.3f}$ of the alerts "
            rf"against the Shapley attribution's ${shap_at:.3f}$---a map that "
            rf"explains nothing accepted on ${const_at / shap_at:.1f}$ times as "
            rf"many alerts as one that does")
    else:
        reading = (
            rf"on this slice the control never overtakes the Shapley "
            rf"attribution; its largest excess, at $\tau_d={tau_star:g}$, is "
            rf"${excess:.3f}$")
    # How far the comparison the figure draws depends on the two functions the
    # figure holds fixed. Only a run that carries the sensitivity sweep can say,
    # so the sentence is omitted rather than guessed when the sweep is absent.
    sweep_clause = ""
    if _SOURCE and "addons" in _SOURCE:
        pairs = _SOURCE["addons"]["sensitivity"]["constant_vs_treeshap"]
        wider = sum(1 for p in pairs if p["constant_wider"])
        sweep_clause = (
            rf", and it is a property of \emph{{this}} $F$ and \emph{{this}} $D$: "
            rf"Section~\ref{{sec:controls}} re-measures the comparison under "
            rf"three instantiations of each and finds the control ahead in "
            rf"{wider} of the {len(pairs)} combinations")
    if reversal is not None:
        reading += (
            rf". The ordering reverses at $\tau_d={reversal:g}$ (${shap[reversal]:.3f}$ "
            rf"against ${const[reversal]:.3f}$), so the acceptance of the "
            rf"degenerate map is confined to a band of the disclosure threshold "
            rf"rather than holding across it")
    return rf"""\begin{{figure}}[t]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[txaiaxis,
  xmin=-0.03, xmax=1.03, ymin=-0.05, ymax=1.16,
  ytick={{0,0.25,0.5,0.75,1}},
  xlabel={{Disclosure threshold $\tau_d$ (at $\tau_f=0.5$, $\tau_b=0.8$)}},
  ylabel={{Fraction of alerts admissible}},
  legend style={{at={{(0.03,0.70)}},anchor=west}}]
{chr(10).join(plots)}
\draw[<->,black!55,line width=0.5pt]
  (axis cs:{tau_star:g},{min(const_at, shap_at) + 0.04:.3g}) --
  (axis cs:{tau_star:g},{max(const_at, shap_at) - 0.04:.3g});
\node[anchor=south,font=\scriptsize,text=black!70,inner sep=2pt]
  at (axis cs:{tau_star:g},{max(const_at, shap_at) + 0.02:.3g})
  {{$\tau_d={tau_star:g}$: ${const_at:.2f}$ vs ${shap_at:.2f}$}};
\end{{axis}}
\end{{tikzpicture}}
\caption{{The four numeric conditions of Definition~\ref{{def:admissible}}, applied
to the alerts of Section~\ref{{sec:numsetup}} as the disclosure threshold is
relaxed with the other two thresholds held at the values used throughout
Section~\ref{{sec:numerical}} (the analyst role; the other three roles coincide
with it to within a percentage point on this slice). This is the measurement
behind the calibration argument: {reading}. The band is a property of this
instantiation, not a theorem{sweep_clause}. What generalizes is therefore not
that the numeric conditions cannot exclude the degenerate map, but that whether
they exclude it is decided by how $F$ and $D$ are instantiated---which is why
\eqref{{eq:calibration}} and the contract must pin that choice down rather than
leave it to the deployment. Single corpus, single detector, single seed.}}
\label{{fig:admissible}}
\end{{figure}}"""


def fig_bridge() -> str:
    """E5/E5b: measured target-action advantage against its two upper bounds."""
    manuscript: List[Tuple[float, float]] = []
    dobrushin: List[Tuple[float, float]] = []
    dropped = 0
    for name in ("treeshap", "lime", "random", "constant"):
        block = load(BRIDGE_FILE)["by_explainer"]
        if name not in block:
            continue
        for row in block[name]["rows"]:
            lhs = row["lhs_adv_tv"]
            if lhs <= 0.0:            # exactly zero on both sides: no log position
                dropped += 1
                continue
            manuscript.append((row["rhs_manuscript"], lhs))
            dobrushin.append((row["rhs_dobrushin"], lhs))
    shown = len(manuscript)
    total = shown + dropped
    return rf"""\begin{{figure}}[t]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[txaiaxis,
  xmode=log, ymode=log, log basis x=10, log basis y=10,
  xmin=1.2e-3, xmax=0.9, ymin=2.5e-5, ymax=0.9,
  xlabel={{Right-hand side of \eqref{{eq:bridge}}}},
  ylabel={{Measured $\mathrm{{Adv}}^{{\mathrm{{TV}}}}_{{E,\Gamma}}$}},
  legend style={{at={{(0.03,0.88)}},anchor=west}}]
\addplot[black,densely dashed,mark=none,domain=1.2e-3:0.9,samples=2,
  forget plot] {{x}};
\addplot[only marks,color=oipurple,mark=o,mark size=1.3pt] coordinates
  {{{coords(manuscript)}}};
\addlegendentry{{generic constant $L_\pi=1$}}
\addplot[only marks,color=oiblue,mark=x,mark size=1.9pt] coordinates
  {{{coords(dobrushin)}}};
\addlegendentry{{Dobrushin coefficient $\delta$}}
\end{{axis}}
\end{{tikzpicture}}
\caption{{Theorem~\ref{{thm:bridge}} on the instantiated kernel. Each of the
${shown}$ plotted configurations lies below the diagonal $y=x$ (dashed), so the
inequality is not violated, and each lies far below it, so the bound is
loose. Replacing the generic constant $L_\pi=1$ by the ergodic
coefficient \eqref{{eq:dobrushin}} moves each point left without moving it above
the diagonal, which is the sharpening claimed in the text. The remaining ${dropped}$ of the ${total}$ recomputed
configurations are the constant control's, omitted because both sides are
exactly zero and so have no position on logarithmic axes; they are also the only
configurations in which the bound is attained. Points are configurations, not alerts, and
holding on the configurations of one instantiation is not a claim that the
bound is tight or loose in general.}}
\label{{fig:bridge}}
\end{{figure}}"""


def table_e8() -> str:
    """E8: equilibrium release width and defender value over the weight sweep.

    A table rather than a figure: the quantity of interest is a lattice level,
    which takes five values on a grid of eighteen settings, and a plot of a
    step function over two swept weights would carry less than the numbers do.
    """
    data = load("E8_release_game")
    etas = sorted({row["eta"] for row in data["sensitivity"]})
    rhos = sorted({row["rho"] for row in data["sensitivity"]})
    cell = {(row["rho"], row["eta"]): row for row in data["sensitivity"]}

    def width_at(rho: float, eta: float) -> int:
        return int(cell[(rho, eta)]["pure_defender_strategy"]
                   .split(",")[0].split("=")[1])

    # The caption makes three claims about the shape of this table. They held on
    # the corpus the table was first written for; on a re-run they are claims
    # about numbers nobody has looked at, so they are checked here rather than
    # trusted.
    for rho in rhos:
        widths = [width_at(rho, e) for e in etas]
        if widths != sorted(widths):
            raise SystemExit(f"caption claims the release width is nondecreasing "
                             f"in eta; at rho={rho:g} it reads {widths}")
    for eta in etas:
        widths = [width_at(r, eta) for r in rhos]
        if widths != sorted(widths, reverse=True):
            raise SystemExit(f"caption claims the release width is nonincreasing "
                             f"in rho; at eta={eta:g} it reads {widths}")
    smallest = min(width_at(r, e) for r in rhos for e in etas)
    if any(width_at(r, etas[0]) != smallest for r in rhos):
        raise SystemExit("caption claims the least release width at the smallest "
                         "eta for every rho; the sweep disagrees")
    attacker = {row["attacker_strategy"].split("|")[0] for row in
                data["sensitivity"]}
    if attacker != {"M4_observe"}:
        raise SystemExit(f"caption claims pure observation at every setting; "
                         f"the sweep plays {sorted(attacker)}")

    lines: List[str] = []
    for rho in rhos:
        parts = [f"${rho:g}$"]
        for eta in etas:
            row = cell[(rho, eta)]
            parts.append(f"${width_at(rho, eta)}$")
            parts.append(f"${row['defender_utility']:.3f}$")
        lines.append(" & ".join(parts) + r"\\")
    body = "\n".join(lines)

    n_d, n_a = len(data["defender_strategies"]), len(data["attacker_strategies"])
    worst = max(abs(row["randomisation_gain"]) for row in data["sensitivity"])
    # Round the mantissa *up*, not to nearest: the caption states this figure as
    # a bound the gain does not exceed, and `.0e` turned 1.110e-16 into
    # "1e-16" -- a bound the measurement it summarises violates.
    expo = math.floor(math.log10(worst))
    mant = math.ceil(worst / 10 ** expo * 10) / 10
    gap = rf"{mant:.1f}\times10^{{{expo}}}"
    header = " & ".join(rf"\multicolumn{{2}}{{c}}{{$\eta={e:g}$}}" for e in etas)
    sub = " & ".join(r"$k^\star$ & $U_D$" for _ in etas)
    return rf"""\begin{{table}}[t]
\caption{{Strong Stackelberg equilibrium of the ${n_d}\times{n_a}$ release game
of Section~\ref{{sec:numgame}}, over the two weights the utilities of
Definition~\ref{{def:game}} leave free: $\eta$ on evidence failure and $\rho$ on
disclosure. $k^\star$ is the equilibrium release width on the lattice
$\{{5,10,20,50,100\}}$ and $U_D$ the defender's equilibrium value. The width is
nondecreasing in $\eta$ and nonincreasing in $\rho$: an evidence obligation is
the only term that pays for release, and at $\eta=0$ the equilibrium releases
the least the lattice allows at every $\rho$. The equilibrium is pure at all
${len(data['sensitivity'])}$ settings---the gain from randomizing does not
exceed ${gap}$---and the attacker plays pure observation
($M_4$) at every one of them.}}
\label{{tab:numgame}}
\centering
\footnotesize
\begin{{tabular}}{{@{{}}l{'cc' * len(etas)}@{{}}}}
\toprule
& {header}\\
$\rho$ & {sub}\\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None,
                        help="nested result file from run_numerical_instantiation.py, without "
                             "the .json suffix; default reads the per-stage "
                             "files run_scoring_layer.py writes")
    parser.add_argument("--bridge", default="E5b_kernel_constant",
                        help="the sharpening file matching --source")
    args = parser.parse_args()
    if args.source:
        use_source(args.source)
    global BRIDGE_FILE
    BRIDGE_FILE = args.bridge

    blocks = (fig_deletion(), fig_robustness(), fig_admissible(), fig_bridge(),
              table_e8())
    sys.stdout.write("\n\n".join(blocks) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
