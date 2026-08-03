"""Measurement package for the T-XAI scoring layer.

Scope, stated once so no downstream write-up can drift from it: this package
measures F, B, D, the Equation (7) calibration condition, and both sides of the
Theorem 1 bridge. It does **not** touch the governance layer -- provenance
(Omega), the claim-evidence graph, the release lattice with real consequences,
actionability (A_Gamma / Psi), or threat class M_3 -- because BODMAS carries no
telemetry provenance, no analyst roles, and no response playbook.
"""

from __future__ import annotations

from .config import ExperimentConfig
from .data import Dataset, load_bodmas, temporal_split, verify_feature_layout
from .detectors import build_detector, score_function
from .explainers import build_explainer
from .metrics import (attribution_distance, deletion_curve, disclosure_D,
                      faithfulness_F,
                      is_admissible, normalise_attribution, robustness_B,
                      sufficiency_Phi, summarise)
from .perturbations import apply_perturbation, check_invariants
from .release import (ResponseKernel, adv_tv, bridge_check,
                      role_disclosure_profile, check_view_monotonicity,
                      sharp_kernel)

__all__ = [
    "ExperimentConfig",
    "Dataset", "load_bodmas", "temporal_split", "verify_feature_layout",
    "build_detector", "score_function", "build_explainer",
    "normalise_attribution", "attribution_distance", "robustness_B",
    "faithfulness_F", "sufficiency_Phi", "disclosure_D", "is_admissible",
    "deletion_curve",
    "summarise",
    "apply_perturbation", "check_invariants",
    "ResponseKernel", "sharp_kernel", "adv_tv", "bridge_check",
    "role_disclosure_profile", "check_view_monotonicity",
]
