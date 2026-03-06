# insurance-cv

Temporal cross-validation for insurance pricing models. Walk-forward splits that respect policy year, accident year, and IBNR development structure.

```bash
uv pip install insurance-cv
```

---

## The problem with standard k-fold in insurance

K-fold cross-validation randomly partitions data into folds. For insurance pricing, this is wrong in at least three ways.

**Temporal leakage.** Insurance claims develop over time. A motor claim reported 18 months after the accident may still be open. If you train on 2022 data and test on 2020 data, your model sees future development patterns that wouldn't have been available at the 2020 pricing date. K-fold does this routinely.

**IBNR contamination.** For any accident date near your training cutoff, some claims will not yet be reported or fully developed (Incurred But Not Reported). If those claims appear in your training set, the model learns from targets that are systematically understated. The fix is a development buffer - exclude claims with accident dates in the N months before your test window from both training and test sets.

**Seasonal confounding.** Motor claims peak in winter. Property claims follow weather cycles. If a randomly-selected test fold contains a disproportionate share of December policies, the test loss will look different to what you'd see prospectively. A prospective evaluation should test on a contiguous future period with the same seasonal mix the model will face in deployment.

The result of using k-fold on insurance data is a model that looks better in CV than it performs in the rating year. Prospective monitoring then shows a gap between modelled and actual loss ratios that is partly attributable to the leaky evaluation methodology.

---

## How this library fixes it

All splits in `insurance-cv` are **walk-forward** (or boundary-aligned): training data always precedes test data in calendar time, with a configurable gap for IBNR development.

Three split generators cover the main use cases:

| Function | When to use it |
|---|---|
| `walk_forward_split` | General-purpose. Expanding training window, rolling test. Standard choice for motor, home, commercial. |
| `policy_year_split` | When rate changes align to policy year boundaries and you want clean PY-aligned folds. |
| `accident_year_split` | Long-tail lines (liability, PI) where accident year development varies across the triangle. |

All generators return `TemporalSplit` objects and yield `(train_idx, test_idx)` tuples that index into your DataFrame. They are also wrapped by `InsuranceCV`, which implements the sklearn `BaseCrossValidator` interface so you can pass them directly to `GridSearchCV`, `cross_val_score`, etc.

---

## Quickstart

```python
import pandas as pd
from insurance_cv import walk_forward_split
from insurance_cv.diagnostics import temporal_leakage_check, split_summary
from insurance_cv.splits import InsuranceCV

# df has an 'inception_date' column and several years of policy data
splits = walk_forward_split(
    df,
    date_col="inception_date",
    min_train_months=18,    # need at least 1.5 years to cover seasonality
    test_months=6,          # evaluate on 6-month windows
    step_months=6,          # non-overlapping test periods
    ibnr_buffer_months=3,   # exclude claims in the 3 months before each test window
)

# Always validate before running the model
check = temporal_leakage_check(splits, df, date_col="inception_date")
if check["errors"]:
    raise RuntimeError("\n".join(check["errors"]))

print(split_summary(splits, df, date_col="inception_date"))
# fold  train_n  test_n  train_end   test_start  gap_days
#    1     2841     957  2019-12-31  2020-04-01        91
#    2     4189    1002  2020-06-30  2020-10-01        93
#  ...

# sklearn-compatible: pass to cross_val_score or GridSearchCV
from sklearn.model_selection import cross_val_score
cv = InsuranceCV(splits, df)
scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_poisson_deviance")
```

---

## API

### `walk_forward_split`

```python
walk_forward_split(
    df: pd.DataFrame,
    date_col: str,
    min_train_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
    ibnr_buffer_months: int = 3,
) -> list[TemporalSplit]
```

Generates an expanding-window walk-forward split. The earliest data is always included in training. Each fold advances the test window by `step_months`. The IBNR buffer excludes rows in the `ibnr_buffer_months` months before `test_start` from both train and test.

Setting `step_months == test_months` gives non-overlapping test windows (the usual choice for insurance). Smaller values increase fold count but introduce correlation between adjacent test periods.

For long-tail lines, `ibnr_buffer_months` should be 12–24 months. For motor it is typically 3–6 months.

### `policy_year_split`

```python
policy_year_split(
    df: pd.DataFrame,
    date_col: str,
    n_years_train: int,
    n_years_test: int = 1,
    step_years: int = 1,
) -> list[TemporalSplit]
```

Splits aligned to 1 Jan – 31 Dec policy year boundaries. Use this when your rate changes are annual and you want clean year-aligned train/test boundaries. There is no IBNR buffer because the year boundary is treated as a natural development cutoff - if you need one, adjust `n_years_train` to leave a gap year.

### `accident_year_split`

```python
accident_year_split(
    df: pd.DataFrame,
    date_col: str,
    development_col: str,
    min_development_months: int = 12,
) -> list[TemporalSplit]
```

Generates one fold per accident year, filtering out years where median claim development is below `min_development_months`. The `development_col` should contain months from accident date to valuation date. This is the right approach for liability and professional indemnity where the development triangle matters.

### `TemporalSplit`

```python
TemporalSplit(
    date_col: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    ibnr_buffer_months: int = 0,
    label: str = "",
)
```

A single split definition. Call `.get_indices(df)` to get `(train_idx, test_idx)` as numpy integer arrays.

### `InsuranceCV`

```python
InsuranceCV(splits: list[TemporalSplit], df: pd.DataFrame)
```

Wraps a list of `TemporalSplit` objects as a sklearn-compatible CV splitter. Implements `split()` and `get_n_splits()`. Pass to `cross_val_score`, `GridSearchCV`, or any other sklearn utility that accepts a CV splitter.

### `temporal_leakage_check`

```python
temporal_leakage_check(
    splits: list[TemporalSplit],
    df: pd.DataFrame,
    date_col: str,
) -> dict[str, list[str]]
```

Returns `{"errors": [...], "warnings": [...]}`. Run this before any model fitting. An empty `errors` list means no temporal leakage was detected.

### `split_summary`

```python
split_summary(
    splits: list[TemporalSplit],
    df: pd.DataFrame,
    date_col: str,
) -> pd.DataFrame
```

Returns a DataFrame with one row per fold: fold number, train/test sizes, actual date boundaries, gap days, and IBNR buffer months. Useful for confirming that your splits look sensible before committing compute to model fitting.

---

## IBNR buffer: choosing the right value

The IBNR buffer is the most consequential parameter in `walk_forward_split`. A buffer that is too short means partially-developed claims contaminate your test evaluation; too long reduces the amount of usable test data.

Rough guidelines by line:

| Line | Typical buffer |
|---|---|
| Motor own damage | 3–6 months |
| Motor third party property | 6–12 months |
| Motor third party bodily injury | 12–24 months |
| Home buildings | 6–12 months |
| Employers' liability | 24–36 months |
| Professional indemnity | 24–48 months |

These are starting points. The right value depends on your claims handling speed, the proportion of large/complex claims, and how you define your loss target (paid vs. incurred vs. ultimate).

---

## Development

```bash
git clone https://github.com/burningcost/insurance-cv
cd insurance-cv
uv sync --dev
uv run pytest -v
```

Tests are designed to run on Databricks (serverless) for the compute-heavy cases. On a local machine `uv run pytest -v` covers the full test suite in seconds since the fixtures use synthetic data.

---

## Licence

MIT. See [LICENSE](LICENSE).
