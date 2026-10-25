"""Unit tests for the Phase 1 cleaning pass."""

from pathlib import Path

import pandas as pd

from house_price.cleaning import phase1_clean
from house_price.loading import load_csv

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_bengaluru.csv"


def test_sample_cleans_6_rows_to_5():
    df = load_csv(SAMPLE)
    clean, report = phase1_clean(df)
    assert report["rows_in"] == 6
    assert report["rows_out"] == 5
    assert report["unparseable_sqft"] == 1  # the row with empty total_sqft
    assert report["dropped_unparseable"] == 1
    assert len(clean) == 5


def test_bath_null_filled_with_median():
    df = pd.DataFrame(
        {
            "location": ["A", "A", "B"],
            "size": ["2 BHK", "3 BHK", "2 BHK"],
            "price": [10.0, 20.0, 30.0],
            "bath": [2.0, None, 3.0],
            "total_sqft": ["1000", "1200", "900"],
        }
    )
    clean, report = phase1_clean(df)
    assert report["bath_fill_median"] == 2.5
    assert clean["bath"].isna().sum() == 0


def test_parsed_columns_present():
    df = load_csv(SAMPLE)
    clean, _ = phase1_clean(df)
    assert "total_sqft" in clean.columns
    assert "bhk" in clean.columns
    # 34.46 Sq. Meter must be converted, not kept as 34.46.
    sqm_row = clean[clean["location"] == "Electronic City Phase II"]
    assert abs(sqm_row["total_sqft"].iloc[0] - 34.46 * 10.7639) < 0.01
