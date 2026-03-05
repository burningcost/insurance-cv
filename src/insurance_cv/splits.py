"""
Core split generators for insurance temporal cross-validation.

Design principles
-----------------
- All splits are deterministic and reproducible given the same data.
- IBNR buffers are enforced as hard constraints, not suggestions.
- Yielded (train_idx, test_idx) are numpy integer arrays indexing into the
  original DataFrame passed to split(), consistent with sklearn convention.
- The sklearn CV interface (split / get_n_splits) is implemented so these
  splitters can be passed directly to GridSearchCV, cross_val_score, etc.

Temporal structure in insurance
--------------------------------
Insurance data has three time axes that are easy to conflate:

  - Policy inception date: when cover started. Determines which rating factors
    apply and which experience year the policy belongs to.
  - Accident date: when the loss event occurred. This is what IBNR (Incurred
    But Not Reported) buffers protect against — recent accident dates have
    incomplete claim development.
  - Reporting / valuation date: when the snapshot was taken.

Standard k-fold ignores all of these. It is wrong for insurance because:

  1. Claims reported after the train cutoff leak information about events that
     technically occurred in the train period.
  2. Seasonality (weather, holiday periods) means random folds will mix summer
     and winter in both train and test, making the model look better than it
     deserves on a real prospective test.
  3. Policy year boundaries matter for rate changes — training on mixed rate
     years without accounting for that structure inflates apparent lift.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Generator, Iterator

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


@dataclass
class TemporalSplit:
    """
    A single train/test split defined by calendar boundaries.

    Parameters
    ----------
    date_col : str
        Name of the date column used to assign rows to train or test.
    train_start : pd.Timestamp
        Inclusive start of the training window.
    train_end : pd.Timestamp
        Inclusive end of the training window.
    test_start : pd.Timestamp
        Inclusive start of the test window.
    test_end : pd.Timestamp
        Inclusive end of the test window.
    ibnr_buffer_months : int
        Number of months excluded between train_end and test_start to allow
        claim development. When > 0, any row with date in
        (train_end, test_start) is excluded from both sets.
    label : str
        Human-readable label for this fold (e.g. "2023-Q1").
    """

    date_col: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    ibnr_buffer_months: int = 0
    label: str = ""

    def __post_init__(self) -> None:
        if self.train_end >= self.test_start:
            raise ValueError(
                f"train_end ({self.train_end}) must be before test_start ({self.test_start}). "
                "Temporal leakage would occur."
            )

    def get_indices(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (train_indices, test_indices) as integer index arrays.

        Parameters
        ----------
        df : pd.DataFrame
            The full dataset. Must contain self.date_col.

        Returns
        -------
        train_idx : np.ndarray of int
        test_idx : np.ndarray of int
        """
        dates = pd.to_datetime(df[self.date_col])
        train_mask = (dates >= self.train_start) & (dates <= self.train_end)
        test_mask = (dates >= self.test_start) & (dates <= self.test_end)
        return (
            np.where(train_mask)[0],
            np.where(test_mask)[0],
        )


def _months_offset(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    """Add an integer number of months to a Timestamp."""
    return ts + pd.DateOffset(months=months)


def _period_end(ts: pd.Timestamp) -> pd.Timestamp:
    """Return the last moment of the month containing ts."""
    return (ts + pd.offsets.MonthEnd(0)).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)


def _period_start(ts: pd.Timestamp) -> pd.Timestamp:
    """Return the first moment of the month containing ts."""
    return ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def walk_forward_split(
    df: pd.DataFrame,
    date_col: str,
    min_train_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
    ibnr_buffer_months: int = 3,
) -> list[TemporalSplit]:
    """
    Generate walk-forward (expanding window) temporal splits.

    The training window grows with each fold — the oldest data is always
    included. This is the standard choice for insurance pricing because the
    underlying risk doesn't change so fast that old data is harmful; using
    an expanding window gives more stable frequency and severity estimates.

    An IBNR buffer of ``ibnr_buffer_months`` months is inserted between the
    last training month and the first test month. Rows in that gap are excluded
    from both train and test to avoid using partially-developed claims as
    either training targets or test targets.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``date_col``.
    date_col : str
        Date column (policy inception date or accident date).
    min_train_months : int
        Minimum number of months required before the first fold is generated.
        Default 12 — less than a full year is rarely enough to capture
        seasonality in motor or home lines.
    test_months : int
        Width of each test window in months.
    step_months : int
        How far forward to advance test_start between folds.
        Setting step_months == test_months gives non-overlapping test windows
        (the usual choice). Smaller values increase fold count and variance.
    ibnr_buffer_months : int
        Gap between train_end and test_start. Claims with accident dates in
        this window are excluded from both sets. For long-tail lines (liability,
        professional indemnity) this should be 12+ months. For motor it is
        typically 3–6 months.

    Returns
    -------
    List of TemporalSplit objects.

    Raises
    ------
    ValueError
        If there is insufficient data for even one fold.
    """
    dates = pd.to_datetime(df[date_col])
    global_start = _period_start(dates.min())
    global_end = _period_end(dates.max())

    first_test_start = _months_offset(global_start, min_train_months + ibnr_buffer_months)
    if first_test_start >= global_end:
        raise ValueError(
            f"Insufficient data for walk-forward split. Need at least "
            f"{min_train_months + ibnr_buffer_months + test_months} months of data; "
            f"got {round((global_end - global_start).days / 30.4)} months."
        )

    splits: list[TemporalSplit] = []
    test_start = first_test_start
    fold = 1

    while True:
        test_end = _months_offset(test_start, test_months) - pd.Timedelta(seconds=1)
        if test_end > global_end:
            break

        train_end = _months_offset(test_start, -ibnr_buffer_months) - pd.Timedelta(seconds=1)
        train_start = global_start

        label = f"fold-{fold:02d} train:{train_start:%Y-%m} to {train_end:%Y-%m} | test:{test_start:%Y-%m} to {test_end:%Y-%m}"

        splits.append(
            TemporalSplit(
                date_col=date_col,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                ibnr_buffer_months=ibnr_buffer_months,
                label=label,
            )
        )

        test_start = _months_offset(test_start, step_months)
        fold += 1

    if not splits:
        raise ValueError(
            "No complete folds could be generated. Check min_train_months, "
            "test_months, ibnr_buffer_months relative to your data range."
        )

    return splits


def policy_year_split(
    df: pd.DataFrame,
    date_col: str,
    n_years_train: int,
    n_years_test: int = 1,
    step_years: int = 1,
) -> list[TemporalSplit]:
    """
    Generate splits aligned to policy years (1 Jan – 31 Dec boundaries).

    Policy-year alignment matters when you have rate changes at year boundaries
    — you don't want the model to see a post-rate-change test period trained on
    pre-rate-change data without a clean year boundary to separate them.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``date_col``.
    date_col : str
        Policy inception date column.
    n_years_train : int
        Number of complete policy years in each training window.
    n_years_test : int
        Number of complete policy years in each test window. Default 1.
    step_years : int
        Years to advance between folds. Default 1.

    Returns
    -------
    List of TemporalSplit objects.
    """
    dates = pd.to_datetime(df[date_col])
    min_year = dates.dt.year.min()
    max_year = dates.dt.year.max()

    splits: list[TemporalSplit] = []
    fold = 1
    train_start_year = min_year

    while True:
        train_end_year = train_start_year + n_years_train - 1
        test_start_year = train_end_year + 1
        test_end_year = test_start_year + n_years_test - 1

        if test_end_year > max_year:
            break

        train_start = pd.Timestamp(f"{train_start_year}-01-01")
        train_end = pd.Timestamp(f"{train_end_year}-12-31 23:59:59")
        test_start = pd.Timestamp(f"{test_start_year}-01-01")
        test_end = pd.Timestamp(f"{test_end_year}-12-31 23:59:59")

        label = (
            f"fold-{fold:02d} PY{train_start_year}-{train_end_year} "
            f"-> PY{test_start_year}-{test_end_year}"
        )

        splits.append(
            TemporalSplit(
                date_col=date_col,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                ibnr_buffer_months=0,  # boundary is already a clean year end
                label=label,
            )
        )

        train_start_year += step_years
        fold += 1

    if not splits:
        raise ValueError(
            f"Cannot generate policy-year splits with {n_years_train} train years "
            f"and {n_years_test} test years from data spanning "
            f"{min_year}–{max_year}."
        )

    return splits


def accident_year_split(
    df: pd.DataFrame,
    date_col: str,
    development_col: str,
    min_development_months: int = 12,
) -> list[TemporalSplit]:
    """
    Generate splits based on accident year with development sufficiency filter.

    In long-tail lines, accident years with less than ``min_development_months``
    of observed development should not be used as test targets — you'd be
    evaluating the model on claims that are largely IBNR. This splitter
    identifies which accident years have sufficient development and generates
    one fold per qualifying test year.

    The ``development_col`` must contain the number of months of development
    observed for each claim or policy row (e.g. months from accident date to
    valuation date). Rows where this is below ``min_development_months`` are
    excluded from test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``date_col`` and ``development_col``.
    date_col : str
        Accident date column.
    development_col : str
        Column with months of development observed per row.
    min_development_months : int
        Minimum development needed for an accident year's data to be used as
        a test target. Default 12 months.

    Returns
    -------
    List of TemporalSplit objects.
    """
    dates = pd.to_datetime(df[date_col])
    development = pd.to_numeric(df[development_col], errors="coerce")

    accident_years = sorted(dates.dt.year.unique())

    # For each accident year, check whether the median development is sufficient
    qualified_test_years: list[int] = []
    for yr in accident_years:
        yr_mask = dates.dt.year == yr
        median_dev = development[yr_mask].median()
        if median_dev >= min_development_months:
            qualified_test_years.append(yr)

    if len(qualified_test_years) < 2:
        raise ValueError(
            f"Need at least 2 accident years with >= {min_development_months} months "
            "development to generate train/test splits. "
            f"Found {len(qualified_test_years)} qualifying year(s): {qualified_test_years}."
        )

    splits: list[TemporalSplit] = []
    # Use all preceding qualified years as train, current year as test
    for i, test_year in enumerate(qualified_test_years[1:], start=1):
        train_years = qualified_test_years[:i]
        train_start_year = train_years[0]
        train_end_year = train_years[-1]

        train_start = pd.Timestamp(f"{train_start_year}-01-01")
        train_end = pd.Timestamp(f"{train_end_year}-12-31 23:59:59")
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year}-12-31 23:59:59")

        label = (
            f"fold-{i:02d} AY{train_start_year}-{train_end_year} -> AY{test_year} "
            f"(min dev: {min_development_months}m)"
        )

        splits.append(
            TemporalSplit(
                date_col=date_col,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                ibnr_buffer_months=0,
                label=label,
            )
        )

    return splits


class InsuranceCV(BaseEstimator):
    """
    sklearn-compatible CV splitter wrapping a list of TemporalSplit objects.

    Use this when you need to pass a splitter object to sklearn utilities such
    as ``cross_val_score`` or ``GridSearchCV``.

    Parameters
    ----------
    splits : list of TemporalSplit
        Pre-computed splits, e.g. from ``walk_forward_split()``.
    df : pd.DataFrame
        The full dataset. Required so ``split()`` can resolve indices.

    Examples
    --------
    >>> cv = InsuranceCV(walk_forward_split(df, "inception_date"), df)
    >>> cross_val_score(model, X, y, cv=cv)
    """

    def __init__(self, splits: list[TemporalSplit], df: pd.DataFrame) -> None:
        self.splits = splits
        self.df = df

    def split(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for s in self.splits:
            yield s.get_indices(self.df)

    def get_n_splits(
        self,
        X: np.ndarray | pd.DataFrame | None = None,
        y: np.ndarray | pd.Series | None = None,
        groups: np.ndarray | None = None,
    ) -> int:
        return len(self.splits)
