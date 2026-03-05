"""
Walk-forward cross-validation for a UK motor pricing model.

This example shows how to use insurance-cv with a LightGBM frequency model
on synthetic motor data. The key point is that the CV results here actually
reflect prospective performance — the test sets are always later than the
training data, and a 3-month IBNR buffer prevents partially-developed claims
from polluting the test evaluation.

Run this on Databricks or any environment with LightGBM installed:
    pip install insurance-cv lightgbm
"""

import numpy as np
import pandas as pd

from insurance_cv import walk_forward_split
from insurance_cv.diagnostics import split_summary, temporal_leakage_check
from insurance_cv.splits import InsuranceCV

# ---------------------------------------------------------------------------
# Synthetic motor dataset
# ---------------------------------------------------------------------------
# In a real project, this would be a policy-level dataset with:
#   - Claim counts (Poisson target for frequency model)
#   - Earned vehicle years (exposure weight)
#   - Rating factors: vehicle age, driver age, NCD, area, etc.
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)
n = 10_000

dates = pd.date_range("2018-01-01", "2023-12-31", freq="D")
inception_dates = pd.to_datetime(rng.choice(dates, size=n, replace=True))

df = pd.DataFrame(
    {
        "inception_date": inception_dates,
        "vehicle_age": rng.integers(0, 15, n),
        "driver_age": rng.integers(17, 80, n),
        "ncd_years": rng.integers(0, 10, n),
        "area_code": rng.choice(["A", "B", "C", "D"], n),
        "earned_vehicle_years": rng.uniform(0.5, 1.0, n),
        "claim_count": rng.poisson(0.08, n),
    }
).sort_values("inception_date").reset_index(drop=True)

print(f"Dataset: {len(df):,} policies from {df['inception_date'].min():%Y-%m} to {df['inception_date'].max():%Y-%m}")
print(f"Overall frequency: {df['claim_count'].sum() / df['earned_vehicle_years'].sum():.4f}")

# ---------------------------------------------------------------------------
# Define splits
# ---------------------------------------------------------------------------

splits = walk_forward_split(
    df,
    date_col="inception_date",
    min_train_months=18,   # Need at least 1.5 years to capture seasonality
    test_months=6,
    step_months=6,
    ibnr_buffer_months=3,  # 3-month buffer — standard for motor
)

print(f"\nGenerated {len(splits)} walk-forward folds\n")

# ---------------------------------------------------------------------------
# Validate splits before running the model
# ---------------------------------------------------------------------------

check = temporal_leakage_check(splits, df, date_col="inception_date")
if check["errors"]:
    raise RuntimeError(f"Temporal leakage detected:\n" + "\n".join(check["errors"]))
if check["warnings"]:
    for w in check["warnings"]:
        print(f"WARNING: {w}")

summary = split_summary(splits, df, date_col="inception_date")
print("Split summary:")
print(summary[["fold", "train_n", "test_n", "train_end", "test_start", "gap_days"]].to_string(index=False))
print()

# ---------------------------------------------------------------------------
# Fit a frequency model per fold
# ---------------------------------------------------------------------------
# Uncomment the LightGBM section if lightgbm is installed.
# This section shows the pattern — InsuranceCV plugs directly into sklearn CV.
# ---------------------------------------------------------------------------

# try:
#     import lightgbm as lgb
#     from sklearn.model_selection import cross_val_score
#
#     features = ["vehicle_age", "driver_age", "ncd_years"]
#     X = df[features].values
#     y = df["claim_count"].values
#
#     cv = InsuranceCV(splits, df)
#
#     model = lgb.LGBMRegressor(
#         objective="poisson",
#         n_estimators=200,
#         learning_rate=0.05,
#         num_leaves=31,
#     )
#
#     scores = cross_val_score(
#         model, X, y,
#         cv=cv,
#         scoring="neg_mean_poisson_deviance",
#     )
#
#     print(f"Mean Poisson deviance across {len(scores)} folds: {-scores.mean():.4f} (+/- {scores.std():.4f})")
#
# except ImportError:
#     print("LightGBM not installed — skipping model fit.")

# ---------------------------------------------------------------------------
# Manual fold iteration (if you need exposure-weighted evaluation)
# ---------------------------------------------------------------------------

fold_results = []
for i, split in enumerate(splits):
    train_idx, test_idx = split.get_indices(df)
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]

    # Baseline: predict overall train frequency for every test row
    train_freq = train["claim_count"].sum() / train["earned_vehicle_years"].sum()
    predicted = test["earned_vehicle_years"] * train_freq
    actual = test["claim_count"]

    mae = np.abs(actual - predicted).mean()
    fold_results.append(
        {
            "fold": i + 1,
            "train_size": len(train),
            "test_size": len(test),
            "train_frequency": train_freq,
            "baseline_mae": mae,
        }
    )

results_df = pd.DataFrame(fold_results)
print("Baseline (constant frequency) results per fold:")
print(results_df.to_string(index=False))
