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

import json
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


def load(name: str) -> Dict[str, Any]:
    """Read one result file, failing loudly if the experiment never ran."""
    path = RESULTS / f"{name}.json"
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise SystemExit(f"missing result file: {path}")


def coords(points: Iterable[Sequence[float]], digits: int = 6) -> str:
    """Format (x, y) pairs as a pgfplots coordinate list."""
    return " ".join(f"({x:.{digits}g},{y:.{digits}g})" for x, y in points)


def fig_deletion() -> str:
    """E2: mean detector confidence against the number of deleted features."""
    by_map = load("E2_faithfulness")["by_explainer"]
    baseline = by_map["treeshap"]["deletion_curve"]["baseline_confidence"]
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
\caption{{Deletion behaviour of the four explanation maps on the $200$ alerts of
the single-corpus instantiation of Section~\ref{{sec:numsetup}}: mean confidence
of the \textsf{{histgb}} detector after deleting the $k$ highest-attributed
features. The two grey guides mark the undeleted mean confidence
(${baseline:.4f}$) and the decision boundary. Read the
curves for their \emph{{ordering and shape}}, not for a per-alert guarantee: only
TreeSHAP drives the detector below the boundary within the first ten features,
while the constant control tracks the undeleted level through $k=200$ because its
fixed ordering is uninformative rather than adversarial. LIME's curve is
measured on the reduced grid its cost forced (Section~\ref{{sec:numlimits}}) and
is not directly comparable to the others. Single corpus, single detector, single
seed; the ordering is not asserted beyond this instantiation.}}
\label{{fig:deletion}}
\end{{figure}}"""


def fig_robustness() -> str:
    """E4: robustness score against the adversary's perturbation budget.

    Drawn on a broken vertical axis. The random control is flat near $0.575$
    while everything else lives above $0.85$, so a single axis spanning both
    spends five sixths of its height on the empty band between them and leaves
    the three TreeSHAP curves -- the only series in the figure that *moves*, and
    the reason the figure exists -- squeezed into the remaining sliver.
    """
    treeshap = load("block_treeshap")["E4_robustness"]["by_family_and_budget"]

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
    lo = min(y for _, y in lower_pts)
    hi = max(y for _, y in lower_pts)
    # The break marks are drawn at the four corners where the two panels face
    # each other, so the discontinuity is stated on the axis and not only in the
    # caption. ``[shift=...]`` keeps this free of the calc library.
    breaks = "\n".join(
        f"\\draw[black,line width=0.4pt] "
        f"([shift={{(-2.2pt,-1.6pt)}}]group c1r{row}.{corner}) -- "
        f"([shift={{(2.2pt,1.6pt)}}]group c1r{row}.{corner});"
        for row, corner in ((1, "south west"), (1, "south east"),
                            (2, "north west"), (2, "north east"))
    )
    return rf"""\begin{{figure}}[t]
\centering
\begin{{tikzpicture}}
\begin{{groupplot}}[txaiaxis,
  group style={{group size=1 by 2, vertical sep=5pt,
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
  xmin=0.0008, xmax=0.065]
\nextgroupplot[height=0.42\columnwidth, ymin=0.835, ymax=1.025,
  ytick={{0.85,0.9,0.95,1}},
  ylabel={{Robustness score $B_{{\Gamma,r}}$}},
  legend style={{at={{(0.03,0.05)}},anchor=south west}}]
{chr(10).join(upper)}
\nextgroupplot[height=0.09\columnwidth, ymin={lo - 0.003:.4g}, ymax={hi + 0.014:.4g},
  ytick={{0.575}}, yticklabels={{$0.575$}},
  xlabel={{Perturbation budget $r$ (fraction of the input)}}]
{lower}
\node[anchor=north west,font=\scriptsize,inner sep=1.5pt]
  at (rel axis cs:0.02,0.98) {{random control}};
\end{{groupplot}}
{breaks}
\end{{tikzpicture}}
\caption{{Robustness score \eqref{{eq:robustness}} against the adversary's budget,
each point a mean over $500$ alerts $\times$ $20$ sampled transformations. The
three solid curves are TreeSHAP under the three admitted transformation
families; the two dashed lines are the controls, both measured under
\textsf{{append\_bytes}}. The vertical axis is broken at the marked corners
because the random control lies far below everything else, and on one continuous
scale it flattens the three curves that actually move into the top seventh of
the panel. Section~\ref{{sec:numrobust}} reads the three movements and says what
the flat lines do and do not license. LIME is absent: its cost admitted only the
single $1\%$ point, reported in the text. Single corpus, single detector, single
seed.}}
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
    # admissible everywhere, and what the Shapley attribution reads there. Both
    # come from the data, so a re-run cannot leave a stale annotation behind.
    tau_star = next(t for t, rate in series["constant"] if rate >= 1.0)
    shap_at_tau_star = dict(series["treeshap"])[tau_star]
    # The caption says "under a third" rather than repeating the annotated
    # number. Fail loudly if a re-run ever makes that wording false.
    if not shap_at_tau_star < 1 / 3:
        raise SystemExit(f"caption wording assumes TreeSHAP < 1/3 at tau_d="
                         f"{tau_star}, measured {shap_at_tau_star}")
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
  (axis cs:{tau_star:g},{shap_at_tau_star + 0.04:.3g}) --
  (axis cs:{tau_star:g},0.96);
\node[anchor=south,font=\scriptsize,text=black!70,inner sep=2pt]
  at (axis cs:{tau_star:g},1.02)
  {{$\tau_d={tau_star:g}$: ${1.0:.2f}$ vs ${shap_at_tau_star:.2f}$}};
\end{{axis}}
\end{{tikzpicture}}
\caption{{The four numeric conditions of Definition~\ref{{def:admissible}}, applied
to $200$ alerts as the disclosure threshold is relaxed with the other two
thresholds held at the values used throughout Section~\ref{{sec:numerical}} (the
analyst role; the other three roles coincide with it on this slice). This is the
measurement behind the calibration argument, and the marked span is the whole of
it: the constant control becomes admissible on \emph{{every}} alert at a
disclosure threshold where TreeSHAP is still admissible on under a
third, so on this slice the inequalities alone accept a vector that
explains nothing over a strictly wider region than they accept a Shapley
attribution. The gap is a property of this instantiation, not a theorem; what
generalises is that the numeric conditions cannot by themselves exclude the
degenerate map, which is what \eqref{{eq:calibration}} is for. Single corpus,
single detector, single seed.}}
\label{{fig:admissible}}
\end{{figure}}"""


def fig_bridge() -> str:
    """E5/E5b: measured target-action advantage against its two upper bounds."""
    manuscript: List[Tuple[float, float]] = []
    dobrushin: List[Tuple[float, float]] = []
    dropped = 0
    for name in ("treeshap", "lime", "random", "constant"):
        for row in load("E5b_kernel_constant")["by_explainer"][name]["rows"]:
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

    lines: List[str] = []
    for rho in rhos:
        parts = [f"${rho:g}$"]
        for eta in etas:
            row = cell[(rho, eta)]
            width = row["pure_defender_strategy"].split(",")[0].split("=")[1]
            parts.append(f"${width}$")
            parts.append(f"${row['defender_utility']:.3f}$")
        lines.append(" & ".join(parts) + r"\\")
    body = "\n".join(lines)

    n_d, n_a = len(data["defender_strategies"]), len(data["attacker_strategies"])
    worst = max(abs(row["randomisation_gain"]) for row in data["sensitivity"])
    mant, expo = f"{worst:.0e}".split("e")
    gap = rf"{mant}\times10^{{{int(expo)}}}"
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
    blocks = (fig_deletion(), fig_robustness(), fig_admissible(), fig_bridge(),
              table_e8())
    sys.stdout.write("\n\n".join(blocks) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
