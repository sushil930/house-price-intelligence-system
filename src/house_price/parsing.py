"""Parsing helpers for the messy Bengaluru housing columns.

This module reuses the corrected parsing logic from the project's
Phase 1 reference implementation, including the unit conversions the
original build-plan sketch got wrong:

* ``"1133 - 1384"`` (a range) -> the midpoint, 1258.5.
* ``"34.46Sq. Meter"`` -> 34.46 * 10.7639 = ~370.9 sqft (NOT 34.46).
* ``"1200Sq. Yards"`` -> 1200 * 9 = 10800 sqft.
* Anything unparseable -> ``numpy.nan`` (counted by callers, never
  silently mis-converted).
"""

import re

import numpy as np
import pandas as pd

SQFT_PER_SQ_METER = 10.7639
SQFT_PER_SQ_YARD = 9.0
SQFT_PER_ACRE = 43560.0


def parse_total_sqft(value):
    """Return total area in square feet, or ``numpy.nan`` if unparseable.

    Handles plain numbers (``"1056"``), ranges (``"1133 - 1384"`` ->
    midpoint) and unit-suffixed values (``"34.46Sq. Meter"``,
    ``"1200Sq. Yards"``, acres). The unit factor is applied to the
    parsed number (or range midpoint), never ignored.
    """
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    low = text.lower()
    if "meter" in low or "sqm" in low or low.endswith("sq. m"):
        factor = SQFT_PER_SQ_METER
    elif "yard" in low:
        factor = SQFT_PER_SQ_YARD
    elif "acre" in low:
        factor = SQFT_PER_ACRE
    else:
        factor = 1.0
    numbers = re.findall(r"\d+\.?\d*", text)
    if not numbers:
        return np.nan
    if len(numbers) >= 2 and "-" in text:
        # A range: take the midpoint, then convert units.
        return (float(numbers[0]) + float(numbers[1])) / 2 * factor
    return float(numbers[0]) * factor


def parse_bhk(size):
    """Extract the bedroom count from a ``size`` string.

    ``"2 BHK"`` -> 2, ``"4 Bedroom"`` -> 4, ``"1 RK"`` -> 1.
    Returns ``numpy.nan`` when no number is present.
    """
    if pd.isna(size):
        return np.nan
    match = re.search(r"(\d+)", str(size))
    return float(match.group(1)) if match else np.nan
