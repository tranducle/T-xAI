"""A minimal, computable A_Gamma -- the fourth admissibility condition.

Definition 3 has four conditions and the experiments report three. The missing
one, actionability, is the only condition with a principled reason to exclude a
constant attribution vector: a release is actionable when it tells the analyst
which response to run, and a vector that is identical on every alert cannot.
Reporting "the numeric conditions do not exclude the degenerate map" while
omitting the condition designed to exclude it overstates what was measured, and
the review said so.

So A_Gamma is instantiated here, minimally and explicitly as a proxy. It is the
product of three factors in [0, 1], each of which a deployment could replace:

* **decisiveness** -- how concentrated the released attribution mass is in one
  component group. An explanation whose mass is spread evenly over all nine
  groups indicates no single playbook entry, and rescaling by the uniform floor
  sends exactly that case to 0.
* **grounding** -- of the released features in the modal group, the fraction on
  which this alert actually deviates from the benign reference population. This
  is the precondition the review asked for, and it is what a fixed vector fails:
  its modal group is the same on every alert, so on alerts where that group is
  unremarkable the named action has no evidence behind it.
* **stability** -- the fraction of the sampled perturbation set that leaves the
  named action unchanged. An explanation that names a different playbook entry
  after a semantics-preserving edit cannot be acted on. Optional, because it
  needs the perturbed attributions E4 computes; 1.0 when they are absent, which
  is the value that makes A_Gamma an *upper* estimate rather than a silent
  omission.

What this is not: a validated operational metric. There is no incident-response
telemetry in either corpus, so no playbook entry here has been checked against
what an analyst would really do. `ACTION_PLAYBOOK` is a stand-in whose only
defended property is that it maps each EMBER component group to a distinct
response. Any admissibility rate computed with it must be reported as measured
under an illustrative playbook -- which is weaker than the other three
conditions, and weaker on purpose.
"""

from __future__ import annotations

import logging
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import GROUP_NAMES
from .metrics import feature_group_index, normalise_attribution

__all__ = [
    "ACTION_PLAYBOOK",
    "ActionabilityParts",
    "actionability_A",
    "modal_group",
    "robust_scale",
]

logger = logging.getLogger(__name__)


#: Component group -> (response the release would trigger, its precondition).
#: Illustrative. Each entry is a plausible triage step for a release whose mass
#: sits in that group; none is drawn from a deployed playbook.
ACTION_PLAYBOOK: Mapping[str, Tuple[str, str]] = {
    "byte_histogram": ("submit to sandbox detonation",
                       "byte distribution departs from the benign population"),
    "byte_entropy": ("test for packing, then unpack and rescan",
                     "entropy profile departs from the benign population"),
    "strings": ("pivot on the extracted strings across the estate",
                "string statistics depart from the benign population"),
    "general": ("check the file's size and signing status",
                "file-level metadata departs from the benign population"),
    "header": ("verify the PE header against the vendor's build",
               "header fields depart from the benign population"),
    "sections": ("inspect the flagged section's permissions and entropy",
                 "section layout departs from the benign population"),
    "imports": ("review the imported API surface for capability",
                "import table departs from the benign population"),
    "exports": ("review the exported symbol set",
                "export table departs from the benign population"),
    "data_directories": ("inspect the referenced directory entries",
                         "directory sizes depart from the benign population"),
}


class ActionabilityParts(dict):
    """The three factors and their product, kept separately for reporting.

    A plain dict subclass so it serialises without a custom encoder, but named
    so a reader of the results file knows the parts are not interchangeable
    with the product.
    """


def robust_scale(background: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Per-feature median absolute deviation of the benign-ish background.

    MAD rather than standard deviation because EMBER columns are heavy-tailed
    and a handful of extreme files would otherwise set the scale for the whole
    column, making every alert look unremarkable.

    The floor matters: EMBER has constant columns, and a zero scale would make
    any nonzero difference an infinite deviation.
    """
    med = np.median(background, axis=0)
    mad = np.median(np.abs(background - med), axis=0)
    return np.maximum(mad, floor)


def modal_group(phi: np.ndarray, top_k: int = 20,
                group_index: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
    """The group carrying the most released mass, and that group's mass share.

    Returns:
        (group id per alert, mass share per alert in [0, 1]).
    """
    gidx = feature_group_index() if group_index is None else group_index
    n = phi.shape[0]
    order = np.argsort(-np.abs(phi), axis=1)[:, :top_k]
    rows = np.repeat(np.arange(n), order.shape[1])
    kept = np.zeros_like(phi)
    kept[rows, order.ravel()] = phi[rows, order.ravel()]
    p = normalise_attribution(kept)

    mass = np.zeros((n, len(GROUP_NAMES)), dtype=np.float64)
    np.add.at(mass.T, gidx, p.T)
    gid = np.argmax(mass, axis=1)
    return gid, mass[np.arange(n), gid]


def actionability_A(X: np.ndarray, phi: np.ndarray, background: np.ndarray,
                    top_k: int = 20,
                    deviation_tol: float = 2.0,
                    phi_perturbed: Optional[Sequence[np.ndarray]] = None,
                    group_index: Optional[np.ndarray] = None
                    ) -> ActionabilityParts:
    """A_Gamma per alert, in [0, 1], with its three factors reported alongside.

    Args:
        X: the alerts, shape (n, d).
        phi: attributions on those alerts, shape (n, d).
        background: rows standing in for the benign population, used only for
            the per-feature scale. The substrate's `background` sample is the
            intended argument.
        top_k: release width, matching the D and Phi conventions.
        deviation_tol: how many robust scale units a feature must depart from
            the background median to count as grounding the named action.
        phi_perturbed: attributions on the perturbed copies from E4, if
            available. Absent means the stability factor is 1.0 and A_Gamma is
            an upper estimate.
        group_index: precomputed feature-to-group map.

    Returns:
        ActionabilityParts with keys `A`, `decisiveness`, `grounding`,
        `stability`, `modal_group` -- each an array of length n except
        `modal_group`, which is the group name per alert.

    Raises:
        ValueError: on a shape mismatch between alerts and attributions, which
            would otherwise silently score one explanation against another
            explanation's alerts.
    """
    if X.shape != phi.shape:
        raise ValueError(f"alerts {X.shape} and attributions {phi.shape} "
                         "must have the same shape")
    gidx = feature_group_index() if group_index is None else group_index
    n_groups = len(GROUP_NAMES)

    gid, share = modal_group(phi, top_k, gidx)

    # Decisiveness: rescaled so that mass spread uniformly over the groups
    # scores 0 rather than 1/|C|.
    floor = 1.0 / n_groups
    decisiveness = np.clip((share - floor) / (1.0 - floor), 0.0, 1.0)

    # Grounding: of the released features that sit in the modal group, the
    # fraction on which this alert departs from the benign population.
    med = np.median(background, axis=0)
    scale = robust_scale(background)
    order = np.argsort(-np.abs(phi), axis=1)[:, :top_k]
    grounding = np.zeros(len(X), dtype=np.float64)
    for i in range(len(X)):
        cols = order[i][gidx[order[i]] == gid[i]]
        if len(cols) == 0:                    # cannot happen: gid comes from
            continue                          # the released mass itself
        z = np.abs(X[i, cols] - med[cols]) / scale[cols]
        grounding[i] = float((z > deviation_tol).mean())

    # Stability: does the named action survive the perturbation set?
    if phi_perturbed:
        agree = np.zeros((len(phi_perturbed), len(X)), dtype=np.float64)
        for j, phi_p in enumerate(phi_perturbed):
            gid_p, _ = modal_group(phi_p, top_k, gidx)
            agree[j] = (gid_p == gid)
        stability = agree.mean(axis=0)
    else:
        stability = np.ones(len(X), dtype=np.float64)
        logger.info("actionability: no perturbed attributions supplied; "
                    "stability fixed at 1.0, so A is an upper estimate")

    return ActionabilityParts(
        A=decisiveness * grounding * stability,
        decisiveness=decisiveness,
        grounding=grounding,
        stability=stability,
        modal_group=np.array([GROUP_NAMES[g] for g in gid]),
    )


def playbook_entry(group: str) -> Dict[str, str]:
    """The illustrative response and precondition for a component group."""
    action, precondition = ACTION_PLAYBOOK[group]
    return {"group": group, "action": action, "precondition": precondition}
