"""EMBER-2018 support: the extractor compatibility layer and the corpus loader.

Kept apart from `txai.data` (which loads BODMAS) because the two corpora reach
the same 2,381-dimensional feature space by different routes. BODMAS ships as a
vectorised archive; EMBER-2018 ships as raw JSONL and must be vectorised here,
under the same hashing convention BODMAS was built with -- see `compat` for how
that convention is pinned, and `tests/test_ember.py` for how it is checked.
"""

from __future__ import annotations

from .compat import ENTRY_NAME_CONVENTIONS, build_extractor
from .loader import load_ember2018

__all__ = ["ENTRY_NAME_CONVENTIONS", "build_extractor", "load_ember2018"]
