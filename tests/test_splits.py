"""
Tests for insurance-cv split generators and diagnostics.

The tests verify the properties that matter in production:
- No temporal overlap between train and test under any configuration
- IBNR buffer is enforced and creates a real gap in the data
- sklearn compatibility so these splitters work with GridSearchCV etc.
- Sensible error handling for edge cases
- Both polars and pandas DataFrames are accepted as input
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from insurance_cv import (
    TemporalSplit,
    walk_forward_split,
    policy_year_split,
    accident_year_split,
)
from insurance_cv.diagnostics import temporal_leakage_check, split_summary
from insurance_cv.splits import InsuranceCV


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_motor_df(
    start: str = "2018-01-01",
    end: str = "2023-12-31",
    n: int = 5000,
    date_col: str = "inception_date",
    seed: int = 42,
) -> pl.DataFrame:
    """Synthetic motor policy dataset as a Polars DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    sampled = pd.to_datetime(rng.choice(dates, size=n, replace=True))
    # Build polars directly
    df = pl.DataFrame(
        {
            date_col: sampled.tolist(),
            "premium": rng.uniform(300, 1200, n).tolist(),
            "claim_count": rng.poisson(0.08, n).tolist(),
        }
    ).sort(date_col)
    return df


def make_motor_df_pandas(
    start: str = "2018-01-01",
    end: str = "2023-12-31",
    n: int = 5000,
    date_col: str = "inception_date",
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic motor policy dataset as a Pandas DataFrame (bridge test)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    sampled = rng.choice(dates, size=n, replace=True)
    df = pd.DataFrame(
        {
            date_col: pd.to_datetime(sampled),
            "premium": rng.uniform(300, 1200, n),
            "claim_count": rng.poisson(0.08, n),
        }
    )
    return df.sort_values(date_col).reset_index(drop=True)


def make_liability_df(
    start: str = "2015-01-01",
    end: str = "2022-12-31",
    n: int = 3000,
    seed: int = 7,
) -> pl.DataFrame:
    """Synthetic liability dataset with development months column."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    sampled = pd.Series(pd.to_datetime(rng.choice(dates, size=n, replace=True)))
    valuation = pd.Timestamp("2023-06-30")
    development_months = ((valuation - sampled) / pd.Timedelta(days=30.4)).astype(int).clip(lower=0)
    return pl.DataFrame(
        {
            "accident_date": sampled.tolist(),
            "development_months": development_months.tolist(),
            "incurred": rng.exponential(5000, n).tolist(),
        }
    )


# ---------------------------------------------------------------------------
# TemporalSplit unit tests
# ---------------------------------------------------------------------------


class TestTemporalSplit:
    def test_basic_index_split(self) -> None:
        df = make_motor_df()
        split = TemporalSplit(
            date_col="inception_date",
            train_start=pd.Timestamp("2018-01-01"),
            train_end=pd.Timestamp("2021-12-31"),
            test_start=pd.Timestamp("2022-01-01"),
            test_end=pd.Timestamp("2022-12-31"),
        )
        train_idx, test_idx = split.get_indices(df)
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_basic_index_split_pandas(self) -> None:
        """Pandas DataFrame input should also work."""
        df = make_motor_df_pandas()
        split = TemporalSplit(
            date_col="inception_date",
            train_start=pd.Timestamp("2018-01-01"),
            train_end=pd.Timestamp("2021-12-31"),
            test_start=pd.Timestamp("2022-01-01"),
            test_end=pd.Timestamp("2022-12-31"),
        )
        train_idx, test_idx = split.get_indices(df)
        assert len(train_idx) > 0
        assert len(test_idx) > 0

    def test_rejects_overlapping_dates(self) -> None:
        with pytest.raises(ValueError, match="must be before test_start"):
            TemporalSplit(
                date_col="inception_date",
                train_start=pd.Timestamp("2020-01-01"),
                train_end=pd.Timestamp("2022-06-30"),
                test_start=pd.Timestamp("2022-01-01"),  # before train_end
                test_end=pd.Timestamp("2022-12-31"),
            )

    def test_no_rows_outside_window(self) -> None:
        df = make_motor_df()
        split = TemporalSplit(
            date_col="inception_date",
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2020-06-30"),
            test_start=pd.Timestamp("2021-01-01"),
            test_end=pd.Timestamp("2021-06-30"),
        )
        train_idx, test_idx = split.get_indices(df)
        # Extract date values via pandas for comparison
        dates = df["inception_date"].to_pandas()
        train_dates = pd.to_datetime(dates.iloc[train_idx])
        test_dates = pd.to_datetime(dates.iloc[test_idx])
        assert train_dates.max() <= pd.Timestamp("2020-06-30")
        assert test_dates.min() >= pd.Timestamp("2021-01-01")


# ---------------------------------------------------------------------------
# walk_forward_split tests
# ---------------------------------------------------------------------------


class TestWalkForwardSplit:
    def test_generates_multiple_folds(self) -> None:
        df = make_motor_df()
        splits = walk_forward_split(
            df,
            date_col="inception_date",
            min_train_months=12,
            test_months=6,
            step_months=6,
            ibnr_buffer_months=3,
        )
        assert len(splits) >= 2

    def test_no_temporal_overlap_any_fold(self) -> None:
        df = make_motor_df()
        splits = walk_forward_split(df, date_col="inception_date")
        result = temporal_leakage_check(splits, df, date_col="inception_date")
        assert result["errors"] == [], f"Leakage detected: {result['errors']}"

    def test_ibnr_buffer_creates_gap(self) -> None:
        df = make_motor_df()
        buffer_months = 3
        splits = walk_forward_split(
            df,
            date_col="inception_date",
            ibnr_buffer_months=buffer_months,
        )
        dates = pd.to_datetime(df["inception_date"].to_pandas())
        for split in splits:
            train_idx, test_idx = split.get_indices(df)
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            max_train = dates.iloc[train_idx].max()
            min_test = dates.iloc[test_idx].min()
            gap_days = (min_test - max_train).days
            # Minimum expected gap: buffer_months * 28 days (conservative month)
            assert gap_days >= buffer_months * 28, (
                f"IBNR buffer not enforced: gap is {gap_days} days, "
                f"expected at least {buffer_months * 28} days. Split: {split.label}"
            )

    def test_training_window_expands(self) -> None:
        """Later folds should have larger training sets (expanding window)."""
        df = make_motor_df()
        splits = walk_forward_split(df, date_col="inception_date")
        train_sizes = [len(s.get_indices(df)[0]) for s in splits]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1], (
                "Training set should grow monotonically in walk-forward split"
            )

    def test_insufficient_data_raises(self) -> None:
        """Only 2 months of data cannot satisfy min_train_months=12."""
        df = make_motor_df(start="2023-01-01", end="2023-02-28", n=100)
        with pytest.raises(ValueError, match="Insufficient data"):
            walk_forward_split(df, date_col="inception_date", min_train_months=12)

    def test_split_summary_shape(self) -> None:
        df = make_motor_df()
        splits = walk_forward_split(df, date_col="inception_date")
        summary = split_summary(splits, df, date_col="inception_date")
        assert isinstance(summary, pl.DataFrame)
        assert len(summary) == len(splits)
        expected_cols = {"fold", "train_n", "test_n", "gap_days"}
        assert expected_cols.issubset(set(summary.columns))

    def test_no_overlap_with_zero_buffer(self) -> None:
        """Even with ibnr_buffer_months=0 there should be no overlap."""
        df = make_motor_df()
        splits = walk_forward_split(
            df, date_col="inception_date", ibnr_buffer_months=0
        )
        result = temporal_leakage_check(splits, df, date_col="inception_date")
        assert result["errors"] == []

    def test_pandas_input_accepted(self) -> None:
        """walk_forward_split should accept pandas DataFrames too."""
        df = make_motor_df_pandas()
        splits = walk_forward_split(df, date_col="inception_date")
        assert len(splits) >= 2
        result = temporal_leakage_check(splits, df, date_col="inception_date")
        assert result["errors"] == []


# ---------------------------------------------------------------------------
# policy_year_split tests
# ---------------------------------------------------------------------------


class TestPolicyYearSplit:
    def test_generates_folds(self) -> None:
        df = make_motor_df()  # 2018–2023
        splits = policy_year_split(df, date_col="inception_date", n_years_train=3)
        assert len(splits) >= 1

    def test_no_temporal_overlap(self) -> None:
        df = make_motor_df()
        splits = policy_year_split(df, date_col="inception_date", n_years_train=3)
        result = temporal_leakage_check(splits, df, date_col="inception_date")
        assert result["errors"] == []

    def test_year_boundaries_correct(self) -> None:
        df = make_motor_df()
        splits = policy_year_split(df, date_col="inception_date", n_years_train=3)
        dates = pd.to_datetime(df["inception_date"].to_pandas())
        for split in splits:
            train_idx, test_idx = split.get_indices(df)
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            max_train = dates.iloc[train_idx].max()
            min_test = dates.iloc[test_idx].min()
            assert max_train.year < min_test.year

    def test_insufficient_data_raises(self) -> None:
        df = make_motor_df(start="2022-01-01", end="2022-12-31", n=500)
        with pytest.raises(ValueError, match="Cannot generate"):
            policy_year_split(df, date_col="inception_date", n_years_train=3)


# ---------------------------------------------------------------------------
# accident_year_split tests
# ---------------------------------------------------------------------------


class TestAccidentYearSplit:
    def test_generates_folds(self) -> None:
        df = make_liability_df()
        splits = accident_year_split(
            df,
            date_col="accident_date",
            development_col="development_months",
            min_development_months=12,
        )
        assert len(splits) >= 2

    def test_no_temporal_overlap(self) -> None:
        df = make_liability_df()
        splits = accident_year_split(
            df,
            date_col="accident_date",
            development_col="development_months",
        )
        result = temporal_leakage_check(splits, df, date_col="accident_date")
        assert result["errors"] == []

    def test_immature_years_excluded(self) -> None:
        """Most recent accident year should not appear as test when dev is low."""
        rng = np.random.default_rng(99)
        base = make_liability_df(end="2022-12-31")
        # Recent rows with only 1 month of development - immature
        recent_dates = pd.date_range("2023-01-01", "2023-06-30", freq="ME")
        recent = pl.DataFrame(
            {
                "accident_date": recent_dates.tolist(),
                "development_months": [1] * 6,
                "incurred": rng.exponential(5000, 6).tolist(),
            }
        )
        df = pl.concat([base, recent])
        splits = accident_year_split(
            df,
            date_col="accident_date",
            development_col="development_months",
            min_development_months=12,
        )
        # 2023 should not appear as a test year in any fold
        for split in splits:
            assert split.test_end.year < 2023, (
                f"Immature accident year 2023 incorrectly included as test: {split.label}"
            )

    def test_insufficient_qualified_years_raises(self) -> None:
        """Only one qualifying year means we can't form a train/test pair."""
        df = pl.DataFrame(
            {
                "accident_date": pd.date_range("2022-01-01", "2022-12-31", freq="ME").tolist(),
                "development_months": [24] * 12,
                "incurred": [10000.0] * 12,
            }
        )
        with pytest.raises(ValueError, match="at least 2 accident years"):
            accident_year_split(
                df,
                date_col="accident_date",
                development_col="development_months",
            )


# ---------------------------------------------------------------------------
# sklearn compatibility tests
# ---------------------------------------------------------------------------


class TestSklearnCompatibility:
    def test_insurance_cv_get_n_splits(self) -> None:
        df = make_motor_df()
        splits = walk_forward_split(df, date_col="inception_date")
        cv = InsuranceCV(splits, df)
        assert cv.get_n_splits() == len(splits)

    def test_insurance_cv_split_yields_arrays(self) -> None:
        df = make_motor_df()
        splits = walk_forward_split(df, date_col="inception_date")
        cv = InsuranceCV(splits, df)
        X_dummy = np.zeros((len(df), 1))
        for train_idx, test_idx in cv.split(X_dummy):
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_insurance_cv_compatible_with_cross_val_score(self) -> None:
        """Verify InsuranceCV works with sklearn's cross_val_score."""
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score

        df = make_motor_df(n=2000)
        splits = walk_forward_split(
            df,
            date_col="inception_date",
            min_train_months=18,
            test_months=6,
            step_months=6,
        )
        cv = InsuranceCV(splits, df)
        X = df.select("premium").to_numpy()
        y = df.select("claim_count").to_numpy().ravel()
        # Should not raise
        scores = cross_val_score(Ridge(), X, y, cv=cv, scoring="neg_mean_squared_error")
        assert len(scores) == len(splits)


# ---------------------------------------------------------------------------
# diagnostics tests
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_leakage_check_clean(self) -> None:
        df = make_motor_df()
        splits = walk_forward_split(df, date_col="inception_date")
        result = temporal_leakage_check(splits, df, date_col="inception_date")
        assert result["errors"] == []

    def test_leakage_check_detects_overlap(self) -> None:
        """Walk-forward splits should be clean when built correctly."""
        df = make_motor_df()
        clean_splits = walk_forward_split(df, "inception_date")
        result = temporal_leakage_check(clean_splits, df, "inception_date")
        assert result["errors"] == [], "Walk-forward splits should be clean"

    def test_split_summary_gap_days_positive(self) -> None:
        df = make_motor_df()
        splits = walk_forward_split(
            df, date_col="inception_date", ibnr_buffer_months=3
        )
        summary = split_summary(splits, df, date_col="inception_date")
        assert isinstance(summary, pl.DataFrame)
        assert (summary["gap_days"] >= 0).all()

    def test_split_summary_returns_polars(self) -> None:
        df = make_motor_df()
        splits = walk_forward_split(df, date_col="inception_date")
        summary = split_summary(splits, df, date_col="inception_date")
        assert isinstance(summary, pl.DataFrame)
