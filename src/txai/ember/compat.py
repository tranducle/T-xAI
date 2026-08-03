"""Restore the hashing semantics EMBER's published vectors were computed under.

`ember/features.py` line 192 passes a bare string to a `FeatureHasher` built with
`input_type="string"`:

    entry_name_hashed = FeatureHasher(50, input_type="string") \
        .transform([raw_obj['entry']]).toarray()[0]

`raw_obj['entry']` is a section name such as `".text"`, so the single sample is a
string rather than an iterable of strings. Every other single-valued call site in
the same file wraps twice — `transform([[raw_obj['coff']['machine']]])` — which
makes line 192 an upstream inconsistency rather than a deliberate choice.

scikit-learn used to accept it and iterate the string, hashing each **character**
as its own feature. Since 1.2 it raises:

    ValueError: Samples can not be a single string. The input must be an
    iterable over iterables of strings.

That leaves a fidelity question, not a style question. The EMBER-2018 vectors,
the BODMAS vectors this project compares against, and the official
`ember_model_2018.txt` were all produced under the old behaviour. Silently
"fixing" line 192 would produce vectors in a different feature space from the
corpus and the model, and nothing downstream would notice: `verify_feature_layout`
checks the byte-histogram blocks, which are unaffected.

So this module reproduces the legacy behaviour instead of correcting it, by
replacing the `FeatureHasher` name inside the vendored module with a subclass that
expands a bare-string sample into its characters. Legacy scikit-learn assigned
each element of a sample a value of 1 and summed duplicates, which is exactly what
the new implementation does for `list(".text")`, so the two agree bin for bin.

Only samples that are bare strings are touched; every other call site in
`ember_features.py` passes a list and is left alone.

`ENTRY_NAME_CONVENTIONS` exposes the alternative reading so the choice can be
settled by measurement rather than by argument — see
`calibrate_ember_convention.py`, which scores both against the authors' own
released model.
"""

from __future__ import annotations

import logging
from typing import Iterable, List

from sklearn.feature_extraction import FeatureHasher

from ..third_party import ember_features

__all__ = ["ENTRY_NAME_CONVENTIONS", "build_extractor", "patch_feature_hasher"]

logger = logging.getLogger(__name__)

#: How to interpret a bare-string sample handed to a string-input FeatureHasher.
#:
#: ``chars``  - legacy scikit-learn: hash each character separately. This is what
#:              the published EMBER and BODMAS vectors were computed under, and is
#:              the default.
#: ``token``  - the reading line 192 probably intended: hash the name as one token.
ENTRY_NAME_CONVENTIONS = ("chars", "token")

_ORIGINAL_HASHER = ember_features.FeatureHasher


def _make_legacy_hasher(convention: str) -> type:
    if convention not in ENTRY_NAME_CONVENTIONS:
        raise ValueError(
            f"convention must be one of {ENTRY_NAME_CONVENTIONS}, got "
            f"{convention!r}")

    class _CompatFeatureHasher(_ORIGINAL_HASHER):  # type: ignore[misc,valid-type]
        """A FeatureHasher that accepts the bare-string sample line 192 passes."""

        def transform(self, raw_X: Iterable) -> "object":
            if self.input_type == "string":
                raw_X = [_expand(x, convention) for x in raw_X]
            return super().transform(raw_X)

    _CompatFeatureHasher.__name__ = f"FeatureHasher_{convention}"
    return _CompatFeatureHasher


def _expand(sample: object, convention: str) -> List[str] | object:
    """Turn a bare-string sample into the iterable the hasher expects."""
    if not isinstance(sample, str):
        return sample
    return list(sample) if convention == "chars" else [sample]


def patch_feature_hasher(convention: str = "chars") -> None:
    """Point the vendored module's `FeatureHasher` name at the compat subclass.

    The vendored file is kept byte-identical to upstream so its SHA-256 stays
    verifiable, which rules out editing line 192 in place. Rebinding the name in
    the module namespace is the smallest change that leaves the file untouched.
    """
    ember_features.FeatureHasher = _make_legacy_hasher(convention)
    logger.debug("FeatureHasher patched for entry-name convention %r",
                 convention)


def build_extractor(convention: str = "chars"):
    """Return a `PEFeatureExtractor` for feature version 2 under `convention`.

    Raises:
        ValueError: if the extractor's width is not the 2381 this project and
            BODMAS both assume.
    """
    from ..config import N_FEATURES

    patch_feature_hasher(convention)
    extractor = ember_features.PEFeatureExtractor(
        feature_version=2, print_feature_warning=False)
    if extractor.dim != N_FEATURES:
        raise ValueError(
            f"extractor yields {extractor.dim} features, not the {N_FEATURES} "
            "this project and the BODMAS matrix both assume")
    return extractor
