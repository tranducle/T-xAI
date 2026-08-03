"""Experiment stages E1-E7 and the ablation grids.

Each stage is a pure function of (config, substrate) returning a JSON-safe dict.
Nothing here decides whether a result is good; stages measure, and the write-up
interprets against the claim matrix in EVIDENCE_GATE.md.

Alert population: every stage operates on the alerts the detector actually
raises on *held-out future* samples -- test-set rows scored >= 0.5. That is the
population the framework is about. Explaining rows the detector never flags
would measure the explainer on inputs no analyst ever sees.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .config import ExperimentConfig, GROUP_NAMES, ROLE_ORDER
from .data import Dataset, temporal_split
from .detectors import build_detector, score_function
from .explainers import build_explainer
from .metrics import (deletion_curve, disclosure_D, faithfulness_F,
                      feature_group_index,
                      is_admissible, robustness_B, sufficiency_Phi, summarise)
from .perturbations import apply_perturbation
from .release import (bridge_check, check_view_monotonicity,
                      role_disclosure_profile, sharp_kernel)

__all__ = ["Substrate", "build_substrate", "build_reference_bank", "e1_detector", "e2_faithfulness",
           "e3_calibration", "e4_robustness", "e5_bridge", "e6_disclosure",
           "e7_admissibility"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Substrate:
    """A fitted detector plus the alert population its explanations describe."""

    detector_name: str
    seed: int
    model: object
    score_fn: object
    X_alerts: np.ndarray
    y_alerts: np.ndarray
    reference: np.ndarray          # (R, d) bank of benign donors
    background: np.ndarray
    train_size: int
    test_size: int
    auc: float
    alert_rate: float
    fit_seconds: float


def build_reference_bank(X_train: np.ndarray, y_train: np.ndarray,
                         score_fn, seed: int, size: int = 5,
                         max_score: float = 0.01) -> np.ndarray:
    """Real training samples that are labelled benign *and* scored benign.

    Both conditions are required. The label alone is not enough: a mislabelled
    or hard benign sample that the detector scores as malware would carry
    malicious evidence into the deletion baseline, and deleting toward it would
    not remove the evidence being measured.

    Raises:
        ValueError: when no sample clears the bar, which means the deletion
            experiment has no valid baseline and must not run.
    """
    benign = np.flatnonzero(y_train == 0)
    if len(benign) == 0:
        raise ValueError("no benign training samples; F and Phi are undefined")
    scores = score_fn(X_train[benign])
    clean = benign[scores < max_score]
    if len(clean) < size:
        raise ValueError(
            f"only {len(clean)} training samples are both labelled benign and "
            f"scored below {max_score}; the deletion baseline would carry "
            "malicious evidence and F/Phi would be uninterpretable")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(clean, size=size, replace=False)
    logger.info("reference bank: %d benign donors, scores max=%.2e",
                size, float(score_fn(X_train[chosen]).max()))
    return X_train[chosen].astype(np.float64)


def build_substrate(data: Dataset, cfg: ExperimentConfig, detector_name: str,
                    seed: int, n_alerts: int) -> Substrate:
    """Fit on the past, alert on the future, and keep the alerts.

    The deletion reference is a bank of real, detector-verified-benign training
    samples. Coordinate-wise summaries were tried first and rejected on
    measurement: see `metrics.faithfulness_F` for the numbers.
    """
    from sklearn.metrics import roc_auc_score

    train_idx, test_idx = temporal_split(data, cfg.test_fraction)
    X_train = data.X[train_idx].astype(np.float32)
    y_train = data.y[train_idx]
    X_test = data.X[test_idx].astype(np.float32)
    y_test = data.y[test_idx]

    model = build_detector(detector_name, seed)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0

    score_fn = score_function(model)
    test_scores = score_fn(X_test)
    auc = float(roc_auc_score(y_test, test_scores))

    alert_mask = test_scores >= 0.5
    alert_idx = np.flatnonzero(alert_mask)
    rng = np.random.default_rng(seed)
    if len(alert_idx) > n_alerts:
        alert_idx = rng.choice(alert_idx, size=n_alerts, replace=False)
    alert_idx.sort()

    reference = build_reference_bank(X_train, y_train, score_fn, seed)
    bg_idx = rng.choice(len(X_train), size=min(1000, len(X_train)), replace=False)

    logger.info("[%s seed=%d] AUC=%.6f alerts=%d/%d fit=%.1fs",
                detector_name, seed, auc, int(alert_mask.sum()), len(X_test),
                fit_seconds)

    return Substrate(
        detector_name=detector_name, seed=seed, model=model, score_fn=score_fn,
        X_alerts=X_test[alert_idx].astype(np.float64),
        y_alerts=y_test[alert_idx],
        reference=reference, background=X_train[bg_idx].astype(np.float64),
        train_size=len(train_idx), test_size=len(test_idx), auc=auc,
        alert_rate=float(alert_mask.mean()), fit_seconds=fit_seconds)


_EXPLAINER_CACHE: Dict[tuple, tuple] = {}


def explain(sub: Substrate, cfg: ExperimentConfig, explainer_name: str,
            X: np.ndarray) -> np.ndarray:
    """Attributions for X under one explainer, with the per-explainer budget.

    The explainer instance is cached per (substrate, name, width). Two reasons,
    one of them a correctness bug found after the first full run:

    1. A stateful explainer must keep its state across calls. Rebuilding
       `random` with the same seed on every call replayed the same draws, so
       the perturbed attributions came back bit-identical to the clean ones and
       the control scored B = 1.000 everywhere -- an artefact of construction,
       not a property of the explanation.
    2. `shap.TreeExplainer` re-parses the whole ensemble on construction, and
       `perturbed_attributions` calls this up to 240 times per block.

    The substrate is held in the cache value so `id(sub)` cannot be recycled
    onto a different substrate while an entry is live.
    """
    key = (id(sub), explainer_name, X.shape[1], cfg.lime_samples)
    entry = _EXPLAINER_CACHE.get(key)
    if entry is None:
        entry = (build_explainer(
            explainer_name, model=sub.model, background=sub.background,
            score_fn=sub.score_fn, seed=sub.seed, n_features=X.shape[1],
            lime_samples=cfg.lime_samples), sub)
        _EXPLAINER_CACHE[key] = entry
    t0 = time.perf_counter()
    phi = entry[0].explain(X)
    logger.info("  %s on %d alerts: %.1fs", explainer_name, len(X),
                time.perf_counter() - t0)
    return phi


# ---------------------------------------------------------------------------
# E1 -- detector substrate and the separability question
# ---------------------------------------------------------------------------

def e1_detector(data: Dataset, cfg: ExperimentConfig, sub: Substrate) -> dict:
    """Record the substrate and characterise the near-perfect separability.

    An AUC of ~1.0 survived the switch from a random to a temporal split, so it
    is not train/test contamination. This stage asks the follow-up question a
    reviewer will ask: how *few* features carry it? If a handful suffice, the
    separability is a collection artefact of BODMAS and every result here is
    conditioned on a detector that is easy to explain.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import HistGradientBoostingClassifier

    train_idx, test_idx = temporal_split(data, cfg.test_fraction)
    X_train, y_train = data.X[train_idx].astype(np.float32), data.y[train_idx]
    X_test, y_test = data.X[test_idx].astype(np.float32), data.y[test_idx]

    single_feature_auc = {}
    for name in GROUP_NAMES:
        from .config import EMBER_GROUPS
        lo, hi = EMBER_GROUPS[name]
        clf = HistGradientBoostingClassifier(max_iter=40, random_state=sub.seed)
        clf.fit(X_train[:, lo:hi], y_train)
        single_feature_auc[name] = float(
            roc_auc_score(y_test, clf.predict_proba(X_test[:, lo:hi])[:, 1]))

    return {
        "detector": sub.detector_name, "seed": sub.seed,
        "train_size": sub.train_size, "test_size": sub.test_size,
        "auc": sub.auc, "alert_rate": sub.alert_rate,
        "fit_seconds": sub.fit_seconds,
        "n_alerts_explained": int(len(sub.X_alerts)),
        "alert_malware_fraction": float((sub.y_alerts == 1).mean()),
        "auc_by_feature_group_alone": single_feature_auc,
    }


# ---------------------------------------------------------------------------
# E2 -- faithfulness, with the random control
# ---------------------------------------------------------------------------

def e2_faithfulness(sub: Substrate, cfg: ExperimentConfig,
                    attributions: Dict[str, np.ndarray]) -> dict:
    """F per explainer, plus the log-spaced curve F saturates on.

    The random control sets the floor F must clear. `deletion_curve` carries the
    resolution: on a detector this separable, the linear schedule's first step
    already deletes far more than the decision needs, so F pins to 1.0 for any
    competent explainer while `median_flip_k` still separates them.
    """
    out = {}
    for name, phi in attributions.items():
        X = sub.X_alerts[:len(phi)]
        F = faithfulness_F(sub.score_fn, X, phi, sub.reference,
                           steps=cfg.deletion_steps)
        out[name] = summarise("F", F)
        out[name]["n"] = int(len(F))
        out[name]["deletion_curve"] = deletion_curve(
            sub.score_fn, X, phi, sub.reference)
    return {"detector": sub.detector_name, "seed": sub.seed, "by_explainer": out}


# ---------------------------------------------------------------------------
# E3 -- is the Equation (7) calibration condition falsifiable, and does it hold
# ---------------------------------------------------------------------------

def e3_calibration(sub: Substrate, cfg: ExperimentConfig,
                   attributions: Dict[str, np.ndarray]) -> dict:
    """Test F >= tau_f against the independent sufficiency criterion Phi.

    Equation (7) asks that the faithfulness score be calibrated to a criterion
    the deployment actually cares about. F and Phi are computed by opposite
    procedures (delete-the-important vs keep-only-the-important), so agreement
    is informative and disagreement is possible -- the unit tests pin that they
    can disagree, which is what stops this stage from being vacuous.
    """
    out = {}
    for name, phi in attributions.items():
        X = sub.X_alerts[:len(phi)]
        F = faithfulness_F(sub.score_fn, X, phi, sub.reference,
                           steps=cfg.deletion_steps)
        rows = {}
        for k in cfg.top_k_grid:
            Phi = sufficiency_Phi(sub.score_fn, X, phi, sub.reference, top_k=k)
            passes_F = F >= cfg.tau_f
            agree = int((passes_F == (Phi == 1)).sum())
            rows[int(k)] = {
                "sufficiency_rate": float(Phi.mean()),
                "F_pass_rate": float(passes_F.mean()),
                "agreement_rate": agree / len(F),
                # Eq. (7) asks for Phi = 1. With a donor bank Phi is a fraction,
                # so there are two readings and reporting only one of them
                # misstates the violation count by a factor of three. Strict:
                # the top-k must reproduce the decision under *every* donor.
                # Total-failure: it reproduces it under none.
                "violations_strict": int((passes_F & (Phi < 1)).sum()),
                "F_passes_but_insufficient": int((passes_F & (Phi == 0)).sum()),
                "sufficient_but_F_fails": int((~passes_F & (Phi == 1)).sum()),
                "phi_eq_1_rate": float((Phi == 1).mean()),
                "phi_eq_0_rate": float((Phi == 0).mean()),
                "phi_median": float(np.median(Phi)),
                "n": int(len(F)),
            }
        out[name] = rows
    return {"detector": sub.detector_name, "seed": sub.seed,
            "tau_f": cfg.tau_f, "by_explainer_and_top_k": out}


# ---------------------------------------------------------------------------
# E4 -- robustness under M_1
# ---------------------------------------------------------------------------

def perturbed_attributions(sub: Substrate, cfg: ExperimentConfig,
                           explainer_name: str, X: np.ndarray,
                           families: Sequence[str], budgets: Sequence[float],
                           n_perturb: int) -> Dict[tuple, List[np.ndarray]]:
    """Explain each perturbed batch once and cache it.

    E4 and E5 need the same perturbed attributions, and explaining is the entire
    cost of this programme. Computing them twice would double a two-hour run for
    no new information.
    """
    cache: Dict[tuple, List[np.ndarray]] = {}
    for family in families:
        for strength in budgets:
            rng = np.random.default_rng(sub.seed)   # same draws per family
            phis = []
            for _ in range(n_perturb):
                X_p = apply_perturbation(family, X, rng, strength)
                phis.append(explain(sub, cfg, explainer_name, X_p))
            cache[(family, strength)] = phis
    return cache


def e4_robustness(sub: Substrate, cfg: ExperimentConfig, explainer_name: str,
                  phi_clean: np.ndarray,
                  cache: Dict[tuple, List[np.ndarray]]) -> dict:
    """B over the perturbation families and budgets (ablations A3 and A5).

    Also reports B as a function of |Delta_z| (ablation A6): B is a supremum, so
    it can only fall as the perturbation set grows. A published B without its
    sample size is therefore not interpretable, and this measures how much of
    the number is an artefact of the budget.
    """
    rows: Dict[str, dict] = {}
    set_size: Dict[str, dict] = {}
    for (family, strength), phis in cache.items():
        B = robustness_B(phi_clean, phis)
        rows.setdefault(family, {})[str(strength)] = {
            **summarise("B", B), "n": int(len(B)), "n_perturb": len(phis)}
        if strength == cfg.perturb_strength:
            sizes = [n for n in cfg.n_perturb_grid if n <= len(phis)]
            set_size[family] = {
                int(n): summarise("B", robustness_B(phi_clean, phis[:n]))
                for n in sizes}
    return {"detector": sub.detector_name, "seed": sub.seed,
            "explainer": explainer_name,
            "by_family_and_budget": rows,
            "by_perturbation_set_size": set_size}


# ---------------------------------------------------------------------------
# E5 -- Theorem 1
# ---------------------------------------------------------------------------

KERNEL_TEMPERATURES = (0.25, 1.0, 4.0)


def e5_bridge(sub: Substrate, cfg: ExperimentConfig, explainer_name: str,
              phi_clean: np.ndarray, y: np.ndarray,
              cache: Dict[tuple, List[np.ndarray]]) -> dict:
    """Both sides of Theorem 1, across families, budgets and kernel sharpness.

    The bound is proved, so `holds` is a self-test on this implementation. The
    quantity of interest is the tightness ratio: a bound that is always three
    orders of magnitude loose is true but tells a deployment nothing.
    """
    rows = []
    for (family, strength), phis in cache.items():
        for temperature in KERNEL_TEMPERATURES:
            kernel = sharp_kernel(seed=sub.seed, temperature=temperature)
            result = bridge_check(phi_clean, phis, y, kernel)
            result.update({"family": family, "strength": strength,
                           "temperature": temperature,
                           "explainer": explainer_name,
                           "detector": sub.detector_name, "seed": sub.seed})
            rows.append(result)
    violations = [r for r in rows if not r["holds"]]
    ratios = [r["tightness_ratio"] for r in rows
              if np.isfinite(r["tightness_ratio"])]
    return {"rows": rows, "n_configurations": len(rows),
            "implementation_violations": len(violations),
            "tightness_ratio_summary": summarise("tightness", np.asarray(ratios))
            if ratios else None}


# ---------------------------------------------------------------------------
# E6 / E7 -- disclosure and admissibility
# ---------------------------------------------------------------------------

def e6_disclosure(sub: Substrate, cfg: ExperimentConfig,
                  attributions: Dict[str, np.ndarray]) -> dict:
    """Equation (10) as a working witness, plus the two monotonicity checks."""
    out = {}
    for name, phi in attributions.items():
        out[name] = {
            "role_profile": role_disclosure_profile(phi, cfg.top_k),
            "view_monotonicity": check_view_monotonicity(phi, cfg.top_k_grid),
            "mean_D_by_role_and_top_k": {
                role: {int(k): float(disclosure_D(phi, role, k).mean())
                       for k in cfg.top_k_grid}
                for role in ROLE_ORDER},
        }
    return {"detector": sub.detector_name, "seed": sub.seed,
            "by_explainer": out}


def e7_admissibility(sub: Substrate, cfg: ExperimentConfig,
                     explainer_name: str, F: np.ndarray, B: np.ndarray,
                     phi: np.ndarray) -> dict:
    """Admissibility rate over the threshold grid, per role.

    Reported as a surface, never at one tuned triple: a single admissibility
    number is a statement about the thresholds, not about the explanations.
    """
    surface = []
    for role in ROLE_ORDER:
        D = disclosure_D(phi, role, cfg.top_k)
        for tau_f in cfg.threshold_grid:
            for tau_b in cfg.threshold_grid:
                for tau_d in cfg.threshold_grid:
                    ok = is_admissible(F, B, D, tau_f, tau_b, tau_d)
                    surface.append({"role": role, "tau_f": tau_f,
                                    "tau_b": tau_b, "tau_d": tau_d,
                                    "rate": float(ok.mean())})
    default = {}
    for role in ROLE_ORDER:
        D = disclosure_D(phi, role, cfg.top_k)
        ok = is_admissible(F, B, D, cfg.tau_f, cfg.tau_b, cfg.tau_d)
        default[role] = {"rate": float(ok.mean()),
                         "mean_D": float(D.mean()),
                         "n": int(len(ok))}
    return {"detector": sub.detector_name, "explainer": explainer_name,
            "seed": sub.seed,
            "default_thresholds": {"tau_f": cfg.tau_f, "tau_b": cfg.tau_b,
                                   "tau_d": cfg.tau_d},
            "at_default": default, "surface": surface}
