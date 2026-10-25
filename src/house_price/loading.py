"""Data loading and first-look summaries (Unit I).

Covers the syllabus items: reading a CSV with Pandas, inspecting shape,
dtypes and missing values, and moving between Pandas and NumPy
(indexing, slicing, reshaping, broadcasting) on the price column.
"""

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "area_type",
    "availability",
    "location",
    "size",
    "society",
    "total_sqft",
    "bath",
    "balcony",
    "price",
]


def load_csv(path):
    """Load a housing CSV and validate the expected columns.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if any required column is missing.
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df


def summarize(df):
    """Return a first-look summary dict (shape, dtypes, missing counts)."""
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_per_column": {c: int(n) for c, n in df.isna().sum().items()},
        "numeric_description": {
            c: {
                "mean": float(v["mean"]),
                "std": float(v["std"]),
                "min": float(v["min"]),
                "max": float(v["max"]),
            }
            for c, v in df.describe().items()
        },
    }


def numpy_price_summary(df):
    """Demonstrate NumPy ops on the price column (Unit I).

    Uses boolean masking, slicing, reshaping and broadcasting. Works on
    any number of rows; the reshaped grid simply uses however many
    prices are available.
    """
    prices = df["price"].to_numpy(dtype=float)
    valid = prices[~np.isnan(prices)]
    result = {
        "count": len(valid),
        "mean": float(valid.mean()) if len(valid) else None,
        "first_price": float(valid[0]) if len(valid) else None,
        "last_prices": [float(v) for v in valid[-10:]],
        "standardised_grid": None,
    }
    tail = valid[-10:]
    if len(tail) >= 2:
        even = tail if len(tail) % 2 == 0 else tail[:-1]
        grid = even.reshape(2, -1)
        std = grid.std()
        if std > 0:
            result["standardised_grid"] = (
                (grid - grid.mean()) / std
            ).round(4).tolist()
    return result


def select_examples(df):
    """Label-based (.loc) and positional (.iloc) selection examples."""
    out = {}
    if "location" in df.columns:
        out["whitefield_loc"] = (
            df.loc[df["location"] == "Whitefield", ["size", "price"]]
            .head()
            .to_dict(orient="records")
        )
    out["first_rows_iloc"] = df.iloc[0:5, [0, 2, 8]].to_dict(orient="records")
    return out
