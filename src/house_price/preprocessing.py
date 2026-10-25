"""Phase 2 preprocessing (Unit II).

Syllabus coverage: measurement levels, label + one-hot/dummy encoding,
null handling, IQR and Z-score outlier detection, mean / variance /
standard deviation / covariance, Gaussian checks, and sampling
techniques.
"""

import pandas as pd
from scipy import stats

# Measurement levels of the dataset's columns (Unit II vocabulary).
MEASUREMENT_LEVELS = {
    "area_type": "nominal",
    "availability": "nominal",
    "location": "nominal",
    "size": "ordinal",  # 1 BHK < 2 BHK < 3 BHK ... once parsed to bhk
    "society": "nominal",
    "total_sqft": "ratio",
    "bath": "ratio",
    "balcony": "ratio",
    "price": "ratio",
}


def add_price_per_sqft(df):
    """Add ``price_per_sqft`` (price is in lakh rupees)."""
    out = df.copy()
    out["price_per_sqft"] = out["price"] * 100000 / out["total_sqft"]
    return out


def iqr_outlier_mask(series):
    """Boolean mask that is True for values inside the IQR fences."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (series >= q1 - 1.5 * iqr) & (series <= q3 + 1.5 * iqr)


def zscore_outlier_mask(series, threshold=4.0):
    """Boolean mask that is True for |z| < ``threshold``.

    A zero-variance series has no outliers by this rule (all values
    are kept) instead of collapsing to NaN z-scores.
    """
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(True, index=series.index)
    z = (series - series.mean()) / std
    return z.abs() < threshold


def apply_outlier_filter(df, column="price_per_sqft"):
    """Keep rows inside BOTH the IQR fences and the Z-score bound.

    Returns (filtered_df, report) with the number of rows removed.
    """
    before = len(df)
    keep = iqr_outlier_mask(df[column]) & zscore_outlier_mask(df[column])
    out = df[keep].copy()
    return out, {"rows_in": int(before), "rows_out": len(out),
                 "removed": int(before - len(out))}


def location_tier_from_median(median_prices):
    """Map each location to an integer tier from its median price.

    Uses quartile bins when possible and falls back to rank-based bins
    when there are too few distinct medians for quartiles (e.g. tiny
    samples), so the function never crashes on small data.
    """
    medians = median_prices.sort_values()
    try:
        tiers = pd.qcut(medians, 4, labels=[0, 1, 2, 3], duplicates="drop")
        if tiers.nunique() >= 2:
            # qcut keeps all 4 categories even when an edge was dropped,
            # so codes can have gaps (e.g. 0, 1, 3). Remap the codes that
            # are actually used to contiguous tiers 0..k-1.
            codes = tiers.cat.codes
            remap = {old: new for new, old in
                     enumerate(sorted(codes.unique()))}
            return codes.map(remap).astype(int)
    except (ValueError, TypeError):
        pass
    # Fallback: rank locations and split into up to 4 equal-ish groups.
    ranks = medians.rank(method="first")
    n = len(medians)
    bins = min(4, n)
    if bins < 2:
        return pd.Series(0, index=medians.index, dtype=int)
    return pd.cut(ranks, bins=bins, labels=list(range(bins)),
                  include_lowest=True).astype(int)


def encode_features(df):
    """One-hot encode ``area_type`` and label-encode location tiers.

    Returns (encoded_df, report). Location tier is derived from the
    median price of each location: 0 = budget ... 3 = luxury.
    """
    out = df.copy()
    report = {}
    out = pd.get_dummies(out, columns=["area_type"], drop_first=True)
    report["area_type_dummies"] = [
        c for c in out.columns if c.startswith("area_type_")
    ]
    medians = out.groupby("location")["price"].median()
    tiers = location_tier_from_median(medians)
    out["location_tier"] = out["location"].map(tiers).astype(int)
    report["location_tier_values"] = sorted(out["location_tier"].unique().tolist())
    return out, report


def gaussian_check(series, max_sample=500, random_state=42):
    """Skewness, kurtosis and a KS test of normality (Unit II/III).

    The KS test runs on a z-scored sample against the standard normal.
    A constant series is reported as non-normal with a reason instead
    of producing NaNs.
    """
    result = {
        "skewness": float(series.skew()),
        "kurtosis": float(series.kurt()),
        "ks_statistic": None,
        "ks_pvalue": None,
        "ks_note": None,
    }
    sample = series.dropna()
    if len(sample) > max_sample:
        sample = sample.sample(max_sample, random_state=random_state)
    if sample.std() == 0 or pd.isna(sample.std()):
        result["ks_note"] = "constant series; KS test not applicable"
        return result
    z = stats.zscore(sample)
    ks = stats.kstest(z, "norm")
    result["ks_statistic"] = float(ks.statistic)
    result["ks_pvalue"] = float(ks.pvalue)
    return result


def sampling_demo(df, n=500, random_state=42):
    """Random and stratified sampling practice (Unit II).

    Returns the sample sizes actually drawn (capped at the data size).
    """
    k = min(n, len(df))
    random_sample = df.sample(k, random_state=random_state) if k else df.iloc[0:0]
    stratified = pd.DataFrame()
    if "area_type" in df.columns and len(df):
        # Stratified: up to 100 rows per area_type group, capped by group size.
        parts = []
        for _, group in df.groupby("area_type"):
            take = min(100, len(group))
            parts.append(group.sample(take, random_state=random_state))
        stratified = pd.concat(parts) if parts else pd.DataFrame()
    return {
        "random_sample_rows": len(random_sample),
        "stratified_sample_rows": len(stratified),
    }
