"""Frozen configuration, EMBER-v2 feature layout, and the role lattice.

The feature layout is not taken on trust. `data.verify_feature_layout` re-checks
the two arithmetic invariants that pin it (both histograms are normalised to sum
exactly to 1) on every run, so a wrong dataset or a re-ordered extractor fails
loudly instead of silently mis-attributing disclosure to the wrong component.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple

__all__ = [
    "EMBER_GROUPS",
    "GROUP_NAMES",
    "ROLES",
    "ROLE_ORDER",
    "ROLE_SENSITIVITY",
    "APPEND_ONLY_GROUPS",
    "ExperimentConfig",
]

# ---------------------------------------------------------------------------
# EMBER feature version 2: 2,381 dimensions, concatenated in extractor order.
# Sizes cross-checked against the sibling project's dataset dossier
# (256+256+255+1280+128+10+62+104 = 2351, plus 15x2 data directories = 2381).
# Order verified numerically -- see data.verify_feature_layout.
# ---------------------------------------------------------------------------
EMBER_GROUPS: Dict[str, Tuple[int, int]] = {
    "byte_histogram": (0, 256),
    "byte_entropy": (256, 512),
    "strings": (512, 616),
    "general_info": (616, 626),
    "header": (626, 688),
    "sections": (688, 943),
    "imports": (943, 2223),
    "exports": (2223, 2351),
    "data_directories": (2351, 2381),
}
GROUP_NAMES: Tuple[str, ...] = tuple(EMBER_GROUPS)

N_FEATURES = 2381
SIZE_FEATURE_INDEX = 616  # GeneralFileInfo[0] = file size in bytes

# Groups a functionality-preserving, append-only edit can move. Appending bytes
# or imports never removes anything, so these are the only channels an M1
# adversary who must keep the binary runnable can drive.
APPEND_ONLY_GROUPS: Tuple[str, ...] = (
    "byte_histogram", "byte_entropy", "general_info", "sections", "imports",
)

# ---------------------------------------------------------------------------
# Role lattice: public <= analyst <= auditor <= designer, ordered by privilege.
# w_R(c) is the SENSITIVITY of disclosing component c TO role R -- the cost the
# defender pays. A less privileged (more public) audience costs more, so the
# weights decrease pointwise as privilege increases.
#
# Note what that does and does not buy. Because the weights are pointwise
# ordered, the *raw* Eq. (10) sum is ordered along this chain for every
# explanation. The *normalised* sum is not: Eq. (10)'s division by
# sum_c w_R(c) rescales per role, so a role's normalised cost carries no
# cross-role meaning. release.role_disclosure_profile measures both.
# Proposition `lp` itself is about component inclusion at a fixed role, which
# is release.check_view_monotonicity.
# ---------------------------------------------------------------------------
ROLE_ORDER: Tuple[str, ...] = ("public", "analyst", "auditor", "designer")

ROLE_SENSITIVITY: Mapping[str, Mapping[str, float]] = {
    # component            public analyst auditor designer
    "byte_histogram":     {"public": 0.30, "analyst": 0.15, "auditor": 0.10, "designer": 0.05},
    "byte_entropy":       {"public": 0.40, "analyst": 0.20, "auditor": 0.12, "designer": 0.05},
    "strings":            {"public": 0.90, "analyst": 0.50, "auditor": 0.30, "designer": 0.10},
    "general_info":       {"public": 0.20, "analyst": 0.10, "auditor": 0.05, "designer": 0.02},
    "header":             {"public": 0.60, "analyst": 0.30, "auditor": 0.18, "designer": 0.06},
    "sections":           {"public": 0.70, "analyst": 0.40, "auditor": 0.25, "designer": 0.08},
    "imports":            {"public": 1.00, "analyst": 0.60, "auditor": 0.35, "designer": 0.12},
    "exports":            {"public": 0.80, "analyst": 0.45, "auditor": 0.28, "designer": 0.10},
    "data_directories":   {"public": 0.50, "analyst": 0.25, "auditor": 0.15, "designer": 0.05},
}
ROLES: Tuple[str, ...] = ROLE_ORDER

# Response actions the defender may take on an alert (finite set U).
ACTIONS: Tuple[str, ...] = ("dismiss", "monitor", "quarantine", "escalate")


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable run configuration. Every experiment reads from one of these."""

    seed: int = 42
    n_seeds: int = 5
    test_fraction: float = 0.2

    # Explanation / metric parameters
    top_k: int = 20               # |comp(z)| for D and the sufficiency test
    deletion_steps: int = 20      # points on the F deletion curve
    n_explain: int = 1000         # alerts explained for F
    n_robust: int = 500           # alerts scored for B
    n_perturb: int = 20           # |Delta_z| samples in the sup of Eq. (6)
    perturb_strength: float = 0.01

    # LIME is ~540x slower per instance than TreeSHAP, so it gets its own budget.
    n_explain_lime: int = 200
    n_robust_lime: int = 100
    lime_samples: int = 2000

    # Admissibility thresholds (Definition 3). Swept, never used as if tuned.
    tau_f: float = 0.5
    tau_b: float = 0.8
    tau_d: float = 0.5

    detectors: Tuple[str, ...] = ("histgb", "xgboost", "randomforest")
    explainers: Tuple[str, ...] = ("treeshap", "lime", "random", "constant")
    perturbations: Tuple[str, ...] = ("append_bytes", "add_imports", "generic")

    threshold_grid: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                         0.6, 0.7, 0.8, 0.9, 1.0)
    top_k_grid: Tuple[int, ...] = (5, 10, 20, 50)
    perturb_budget_grid: Tuple[float, ...] = (0.001, 0.005, 0.01, 0.05)
    n_perturb_grid: Tuple[int, ...] = (1, 5, 10, 20, 50)

    metadata: Mapping[str, str] = field(default_factory=dict)
