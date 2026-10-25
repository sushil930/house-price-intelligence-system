"""Phase 1 cleaning pass (Unit I).

Drops rows missing essential fields, fills numeric nulls, parses the
``size`` and ``total_sqft`` columns, and drops rows whose parsed values
are unusable -- only after counting them, so nothing disappears
silently.
"""


from .parsing import parse_bhk, parse_total_sqft


def phase1_clean(df):
    """Run the Phase 1 cleaning pass.

    Returns:
        (clean_df, report) where ``report`` counts every row dropped
        and why, plus the fill value used for ``bath``.
    """
    report = {"rows_in": len(df)}
    df1 = df.dropna(subset=["location", "size", "price"]).copy()
    report["dropped_missing_essentials"] = int(
        report["rows_in"] - len(df1)
    )
    bath_median = float(df1["bath"].median()) if df1["bath"].notna().any() else 0.0
    df1["bath"] = df1["bath"].fillna(bath_median)
    report["bath_fill_median"] = bath_median

    df1["bhk"] = df1["size"].apply(parse_bhk)
    df1["total_sqft_parsed"] = df1["total_sqft"].apply(parse_total_sqft)
    report["unparseable_sqft"] = int(df1["total_sqft_parsed"].isna().sum())
    report["unparseable_bhk"] = int(df1["bhk"].isna().sum())
    df1 = df1.dropna(subset=["total_sqft_parsed", "bhk"])
    report["rows_out"] = len(df1)
    report["dropped_unparseable"] = int(
        report["rows_in"]
        - report["dropped_missing_essentials"]
        - report["rows_out"]
    )
    # Keep the parsed column under the canonical name from here on.
    df1 = df1.drop(columns=["total_sqft"]).rename(
        columns={"total_sqft_parsed": "total_sqft"}
    )
    return df1, report
