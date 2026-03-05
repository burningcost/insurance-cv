"""
Diagnostics for temporal cross-validation splits.

Validating splits before running a model is not optional — temporal leakage
in a CV setup produces optimistic results that won't hold prospectively, and
the error is silent. These functions make the split structure visible and
checkable.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .splits import TemporalSplit


def temporal_leakage_check(
    splits: list[TemporalSplit],
    df: pd.DataFrame,
    date_col: str,
) -> dict[str, list[str]]:
    """
    Verify that no rows appear in both train and test for any fold.

    Also checks that the maximum date in each train set is strictly before the
    minimum date in each test set, and that the IBNR buffer gap is respected.

    Parameters
    ----------
    splits : list of TemporalSplit
        Splits to validate.
    df : pd.DataFrame
        The full dataset.
    date_col : str
        Date column used to assign rows.

    Returns
    -------
    dict with keys ``"errors"`` and ``"warnings"``, each a list of strings.
    If both lists are empty the splits are clean.

    Raises
    ------
    Does not raise — callers should inspect the return value and decide whether
    to proceed.
    """
    errors: list[str] = []
    warnings_list: list[str] = []
    dates = pd.to_datetime(df[date_col])

    for i, split in enumerate(splits):
        train_idx, test_idx = split.get_indices(df)
        label = split.label or f"fold-{i + 1}"

        # Check for index overlap
        overlap = np.intersect1d(train_idx, test_idx)
        if len(overlap) > 0:
            errors.append(
                f"{label}: {len(overlap)} rows appear in both train and test. "
                "This is temporal leakage."
            )

        if len(train_idx) == 0:
            warnings_list.append(f"{label}: training set is empty.")
            continue
        if len(test_idx) == 0:
            warnings_list.append(f"{label}: test set is empty.")
            continue

        max_train_date = dates.iloc[train_idx].max()
        min_test_date = dates.iloc[test_idx].min()

        # Check train does not extend into test period
        if max_train_date >= min_test_date:
            errors.append(
                f"{label}: latest training date ({max_train_date:%Y-%m-%d}) is "
                f"not before earliest test date ({min_test_date:%Y-%m-%d}). "
                "Temporal leakage."
            )

        # Check IBNR buffer is honoured
        if split.ibnr_buffer_months > 0:
            expected_gap_days = split.ibnr_buffer_months * 28  # conservative
            actual_gap_days = (min_test_date - max_train_date).days
            if actual_gap_days < expected_gap_days:
                warnings_list.append(
                    f"{label}: IBNR buffer may be insufficient. "
                    f"Requested {split.ibnr_buffer_months} months "
                    f"({expected_gap_days} days), actual gap is {actual_gap_days} days. "
                    "Check whether the date column is sparse near the boundary."
                )

    return {"errors": errors, "warnings": warnings_list}


def split_summary(
    splits: list[TemporalSplit],
    df: pd.DataFrame,
    date_col: str,
) -> pd.DataFrame:
    """
    Return a summary DataFrame with one row per fold.

    Columns
    -------
    fold : int
        Fold number (1-indexed).
    label : str
        Human-readable fold label.
    train_n : int
        Number of rows in training set.
    test_n : int
        Number of rows in test set.
    train_start : date
        Earliest date in training set.
    train_end : date
        Latest date in training set.
    test_start : date
        Earliest date in test set.
    test_end : date
        Latest date in test set.
    gap_days : int
        Calendar days between last training row and first test row.
    ibnr_buffer_months : int
        Requested IBNR buffer for this fold.

    Parameters
    ----------
    splits : list of TemporalSplit
    df : pd.DataFrame
    date_col : str

    Returns
    -------
    pd.DataFrame
    """
    dates = pd.to_datetime(df[date_col])
    rows: list[dict] = []

    for i, split in enumerate(splits):
        train_idx, test_idx = split.get_indices(df)

        if len(train_idx) > 0:
            train_dates = dates.iloc[train_idx]
            actual_train_start = train_dates.min()
            actual_train_end = train_dates.max()
        else:
            actual_train_start = pd.NaT
            actual_train_end = pd.NaT

        if len(test_idx) > 0:
            test_dates = dates.iloc[test_idx]
            actual_test_start = test_dates.min()
            actual_test_end = test_dates.max()
        else:
            actual_test_start = pd.NaT
            actual_test_end = pd.NaT

        gap_days: int | None = None
        if pd.notna(actual_train_end) and pd.notna(actual_test_start):
            gap_days = (actual_test_start - actual_train_end).days

        rows.append(
            {
                "fold": i + 1,
                "label": split.label or f"fold-{i + 1}",
                "train_n": len(train_idx),
                "test_n": len(test_idx),
                "train_start": actual_train_start.date() if pd.notna(actual_train_start) else None,
                "train_end": actual_train_end.date() if pd.notna(actual_train_end) else None,
                "test_start": actual_test_start.date() if pd.notna(actual_test_start) else None,
                "test_end": actual_test_end.date() if pd.notna(actual_test_end) else None,
                "gap_days": gap_days,
                "ibnr_buffer_months": split.ibnr_buffer_months,
            }
        )

    return pd.DataFrame(rows)
