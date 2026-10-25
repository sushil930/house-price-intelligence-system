"""Unit tests for parsing (the corrected unit conversions)."""

import numpy as np
import pytest

from house_price.parsing import parse_bhk, parse_total_sqft


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1056", 1056.0),
        (1440, 1440.0),
        ("1133 - 1384", 1258.5),  # range -> midpoint
        ("34.46Sq. Meter", pytest.approx(34.46 * 10.7639)),  # NOT 34.46
        ("1200Sq. Yards", 10800.0),  # x9
        ("1 Acre", 43560.0),
    ],
)
def test_parse_total_sqft(raw, expected):
    assert parse_total_sqft(raw) == expected


@pytest.mark.parametrize("raw", ["not a number", "", None, float("nan")])
def test_parse_total_sqft_unparseable_is_nan(raw):
    assert np.isnan(parse_total_sqft(raw))


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2 BHK", 2.0),
        ("4 Bedroom", 4.0),
        ("1 RK", 1.0),
        ("3 BHK", 3.0),
    ],
)
def test_parse_bhk(raw, expected):
    assert parse_bhk(raw) == expected


@pytest.mark.parametrize("raw", ["Studio", "", None])
def test_parse_bhk_unparseable_is_nan(raw):
    assert np.isnan(parse_bhk(raw))
