"""
insurance-cv: Temporal cross-validation for insurance pricing models.

Walk-forward splits that respect policy year, accident year, and IBNR
development structure. Drop-in replacement for sklearn CV splitters
wherever you need time-aware splits.
"""

from .splits import (
    TemporalSplit,
    walk_forward_split,
    policy_year_split,
    accident_year_split,
)
from .diagnostics import temporal_leakage_check, split_summary

__all__ = [
    "TemporalSplit",
    "walk_forward_split",
    "policy_year_split",
    "accident_year_split",
    "temporal_leakage_check",
    "split_summary",
]

__version__ = "0.1.0"
