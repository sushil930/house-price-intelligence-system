"""Phase 3 EDA and statistical inference (Unit III).

Syllabus coverage: descriptive statistics (central tendency, dispersion,
IQR, skewness, kurtosis), univariate / bivariate / multivariate EDA,
correlation, feature engineering, Bayes' theorem, normal / binomial /
Poisson distributions, the central limit theorem with confidence
intervals, and hypothesis testing (t-test, Z-test, one-way ANOVA).

Every test returns a dict with a ``status`` field: ``"ok"`` when the
test ran, or ``"skipped"`` with a ``reason`` when the data cannot
support it (e.g. a group with a single observation). Skipped tests are
reported, never faked.
"""

import numpy as np
import pandas as pd
from scipy import stats


def descriptive_stats(df):
    """Central tendency, dispersion, IQR, skewness, kurtosis, covariance."""
    price = df["price"]
    numeric = df.select_dtypes(include="number")
    cov = None
    if "bath" in numeric.columns and len(df) >= 2:
        cov = float(numeric[["price", "bath"]].cov().iloc[0, 1])
    return {
        "price_mean": float(price.mean()),
        "price_median": float(price.median()),
        "price_mode": float(price.mode().iloc[0]) if not price.mode().empty else None,
        "price_std": float(price.std()),
        "price_variance": float(price.var()),
        "price_range": [float(price.min()), float(price.max())],
        "price_iqr": float(price.quantile(0.75) - price.quantile(0.25)),
        "price_skewness": float(price.skew()),
        "price_kurtosis": float(price.kurt()),
        "mode_location": str(df["location"].mode().iloc[0])
        if not df["location"].mode().empty else None,
        "cov_price_bath": cov,
    }


def correlation_matrix(df):
    """Pearson correlations over numeric columns with nonzero variance."""
    numeric = df.select_dtypes(include="number")
    varying = numeric.loc[:, numeric.std() > 0]
    if varying.shape[1] < 2:
        return pd.DataFrame()
    return varying.corr()


def engineer_features(df):
    """Feature engineering: room density and a premium-location flag."""
    out = df.copy()
    out["room_density"] = out["bhk"] / (out["total_sqft"] / 1000)
    out["is_premium"] = (out["location_tier"] >= 2).astype(int)
    return out


def bayes_high_price_given_location(df, location="Whitefield"):
    """P(price > median | location) via Bayes' theorem.

    P(high | loc) = P(loc | high) * P(high) / P(loc).
    """
    median = df["price"].median()
    high = df["price"] > median
    p_high = float(high.mean())
    p_loc = float((df["location"] == location).mean())
    if p_loc == 0:
        return {"status": "skipped",
                "reason": f"location {location!r} not present in data"}
    p_loc_given_high = float(
        (df.loc[high, "location"] == location).mean()
    ) if high.any() else 0.0
    return {
        "status": "ok",
        "location": location,
        "median_price": float(median),
        "p_high": p_high,
        "p_location": p_loc,
        "p_location_given_high": p_loc_given_high,
        "p_high_given_location": p_loc_given_high * p_high / p_loc,
    }


def distribution_examples(df):
    """Poisson and binomial examples grounded in the data."""
    loc_counts = df["location"].value_counts()
    p_3bhk = float((df["bhk"] == 3).mean())
    return {
        "poisson_lambda_mean_listings_per_location": float(loc_counts.mean()),
        "poisson_pmf_k5": float(stats.poisson.pmf(5, loc_counts.mean())),
        "p_3bhk": p_3bhk,
        "binomial_pmf_k3_n10": float(stats.binom.pmf(3, 10, p_3bhk)),
    }


def clt_confidence_interval(df, sample_size=200, n_bootstraps=1000,
                            random_state=42):
    """95% CI for the mean price from bootstrap sample means (CLT).

    Bootstrap sampling uses replacement, so it works on any data size.
    """
    rng = np.random.default_rng(random_state)
    prices = df["price"].to_numpy(dtype=float)
    prices = prices[~np.isnan(prices)]
    if len(prices) == 0:
        return {"status": "skipped", "reason": "no price values"}
    k = min(sample_size, len(prices))
    means = [
        float(rng.choice(prices, size=k, replace=True).mean())
        for _ in range(n_bootstraps)
    ]
    return {
        "status": "ok",
        "sample_size": int(k),
        "n_bootstraps": int(n_bootstraps),
        "ci_95": [float(np.percentile(means, 2.5)),
                  float(np.percentile(means, 97.5))],
        "mean_of_means": float(np.mean(means)),
    }


def t_test_bhk_vs_other(df):
    """Two-sample Welch t-test: BHK flats vs other property sizes."""
    is_bhk = df["size"].str.contains("BHK", na=False)
    a = df.loc[is_bhk, "price"].dropna()
    b = df.loc[~is_bhk, "price"].dropna()
    if len(a) < 2 or len(b) < 2:
        return {"status": "skipped",
                "reason": f"need >=2 observations per group "
                          f"(BHK n={len(a)}, other n={len(b)})"}
    res = stats.ttest_ind(a, b, equal_var=False)
    return {"status": "ok", "groups": {"BHK": len(a), "other": len(b)},
            "statistic": float(res.statistic), "pvalue": float(res.pvalue)}


def z_test_location_vs_rest(df, location="Whitefield"):
    """Two-sample Z-test for mean price: one location vs the rest.

    Uses the full-data variances as the known population variances
    (the standard textbook Z-test setup); the assumption is documented
    in the result so the test is not mistaken for a t-test.
    """
    a = df.loc[df["location"] == location, "price"].dropna()
    b = df.loc[df["location"] != location, "price"].dropna()
    if len(a) < 1 or len(b) < 1:
        return {"status": "skipped",
                "reason": f"need >=1 observation per group "
                          f"({location} n={len(a)}, rest n={len(b)})"}
    var_a, var_b = float(a.var(ddof=0)), float(b.var(ddof=0))
    se = np.sqrt(var_a / len(a) + var_b / len(b))
    if se == 0:
        return {"status": "skipped", "reason": "zero standard error"}
    z = (a.mean() - b.mean()) / se
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {"status": "ok", "location": location,
            "assumption": "population variances taken as the full-data variances",
            "statistic": float(z), "pvalue": p}


def anova_top_locations(df, top_n=3):
    """One-way ANOVA of mean price across the top-N locations."""
    top = df["location"].value_counts().head(top_n).index
    groups = [df.loc[df["location"] == loc, "price"].dropna() for loc in top]
    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        return {"status": "skipped",
                "reason": "need >=2 locations with >=2 observations each; "
                          f"group sizes: {[len(g) for g in groups]}"}
    res = stats.f_oneway(*groups)
    return {"status": "ok", "locations": [str(t) for t in top],
            "statistic": float(res.statistic), "pvalue": float(res.pvalue)}


def run_all_stats(df):
    """Run every Unit III analysis and return a structured report."""
    corr = correlation_matrix(df)
    return {
        "descriptive": descriptive_stats(df),
        "correlation": corr.round(4).to_dict() if not corr.empty else {},
        "bayes": bayes_high_price_given_location(df),
        "distributions": distribution_examples(df),
        "clt_ci": clt_confidence_interval(df),
        "t_test": t_test_bhk_vs_other(df),
        "z_test": z_test_location_vs_rest(df),
        "anova": anova_top_locations(df),
    }
