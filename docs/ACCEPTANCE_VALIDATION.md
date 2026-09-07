# Acceptance-validation follow-up

## Evidence lineage

Both official follow-up experiments were generated from Git revision `a43c96c192b3f0b475d887adcb494a75a9017359` with `working_tree_dirty=false`, Python 3.14.5, and NumPy 2.4.6. Data, split, and baseline gates pass before the new evidence is used.

## E10: within-scope explanation-side bridge

- Alerts: 500
- Deterministic explanation-only transformations: 18
- Family/kernel rows: 12
- Input max difference before/after: 0.0
- Detector-score max difference before/after: 0.0
- Labels exactly equal: true
- All generic L1 rows satisfy the implemented inequality: true
- All Dobrushin rows satisfy the implemented inequality: true
- Largest generic LHS/RHS: 0.613437 (group_omission, T=0.25)
- Corresponding LHS / RHS: 0.141257 / 0.230271

Interpretation: E10 is an executable numerical instantiation inside the theorem's declared `Delta_z` scope. It is not additional proof, general empirical validation, or a certificate over arbitrary explanation-channel transformations.

## E11: controlled M3 evidence-channel integrity

- Alerts per bundle attack type: 500
- Bundle corruption types: 7
- Minimum authenticated bundle detection rate: 1.000
- Mean structural detection on well-formed edits: 0.000
- Mean authenticated detection on well-formed edits: 1.000
- Observed clean false-reject rate: 0.000
- Persistent-channel malicious detection rate: 1.000
- Persistent-channel benign acceptance rate: 1.000
- Key-compromise full-retag bypass: true

The persistent channel tests content edit, insertion, truncation, replay, reorder, rollback to a valid prefix, and tag corruption. It also verifies benign canonical reserialization and legitimate append. The authenticator key is modeled outside the attacked channel, and key compromise is an explicit failure boundary rather than a hidden assumption.

Interpretation: E11 supports a controlled M3 claim-evidence integrity claim under an uncompromised trust anchor. It does not establish arbitrary M3 resistance or live-SOC security.

## Scientific gates

- G0 data provenance: PASS
- G1 preprocessing/split integrity: PASS
- G2 baseline sanity: PASS
- G3 E10 mechanism: PASS
- G3 E11 mechanism: PASS_WITH_WARNINGS
- G6 result integrity: PASS_WITH_WARNINGS
- G8 claim permission: PASS_WITH_WARNINGS

G8 is the authoritative wording boundary. Its allowed and forbidden claims are stored in `results/gates/G8_acceptance_claim_permission.json`.
