"""Unit tests for Phase 2 preprocessing (Unit II)."""

import pandas as pd

from house_price.preprocessing import (
    apply_outlier_filter,
    encode_features,
    gaussian_check,
    iqr_outlier_mask,
    location_tier_from_median,
    sampling_demo,
    zscore_outlier_mask,
)


def test_iqr_mask_flags_extreme_value():
    s = pd.Series([10.0, 11.0, 12.0, 11.0, 10.0, 1000.0])
    mask = iqr_outlier_mask(s)
    assert not mask.iloc[-1]
    assert mask.iloc[:5].all()


def test_zscore_mask_constant_series_keeps_all():
    s = pd.Series([5.0, 5.0, 5.0])
    assert zscore_outlier_mask(s).all()


def test_outlier_filter_removes_extreme_price_per_sqft():
    df = pd.DataFrame(
        {
            "price": [40.0, 42.0, 41.0, 43.0, 39.0, 5000.0],
            "total_sqft": [1000.0] * 6,
            "price_per_sqft": [4000.0, 4200.0, 4100.0, 4300.0, 3900.0,
                               500000.0],
        }
    )
    out, report = apply_outlier_filter(df)
    assert report["removed"] == 1
    assert len(out) == 5


def test_location_tiers_are_contiguous():
    medians = pd.Series({"A": 10.0, "B": 50.0, "C": 90.0})
    tiers = location_tier_from_median(medians)
    assert sorted(tiers.unique().tolist()) == list(range(tiers.nunique()))


def test_encode_features_produces_dummies_and_tiers():
    df = pd.DataFrame(
        {
            "location": ["A", "A", "B", "B", "C"],
            "price": [10.0, 12.0, 50.0, 55.0, 90.0],
            "area_type": ["X", "Y", "X", "Y", "X"],
        }
    )
    out, report = encode_features(df)
    assert report["area_type_dummies"] == ["area_type_Y"]
    assert out["location_tier"].tolist() == [0, 0, 1, 1, 2]


def test_gaussian_check_constant_series_reports_note():
    result = gaussian_check(pd.Series([7.0, 7.0, 7.0]))
    assert result["ks_statistic"] is None
    assert result["ks_note"] is not None


def test_sampling_demo_counts():
    df = pd.DataFrame(
        {
            "area_type": ["X", "X", "Y", "Y", "Y"],
            "price": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    result = sampling_demo(df, n=3)
    assert result["random_sample_rows"] == 3
    assert result["stratified_sample_rows"] == 5  # capped by group sizes
