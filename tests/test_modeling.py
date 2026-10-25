"""Unit tests for Phase 5 modelling (Unit V)."""

import numpy as np
import pandas as pd
import pytest

from house_price.modeling import (
    build_features,
    evaluate,
    kmeans_segmentation,
    safe_r2,
    split_data,
    train_multiple,
    train_simple,
    vif_table,
)


def make_frame(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "total_sqft": rng.uniform(500, 2500, n),
            "bath": rng.integers(1, 4, n).astype(float),
            "bhk": rng.integers(1, 5, n).astype(float),
            "balcony": rng.integers(0, 3, n).astype(float),
            "location_tier": rng.integers(0, 3, n).astype(float),
            "room_density": rng.uniform(1, 4, n),
            "is_premium": rng.integers(0, 2, n).astype(float),
            "price": rng.uniform(20, 150, n),
        }
    )


def test_build_features_drops_missing_rows():
    df = make_frame(5)
    df.loc[0, "bath"] = np.nan
    X, _y, features, dropped = build_features(df)
    assert dropped == 1
    assert len(X) == 4
    assert "total_sqft" in features


def test_split_sizes_cover_all_rows():
    df = make_frame(20)
    X, y, _, _ = build_features(df)
    Xtr, Xv, Xte, _ytr, _yv, _yte = split_data(X, y)
    assert len(Xtr) + len(Xv) + len(Xte) == 20
    assert len(Xtr) >= 1 and len(Xv) >= 1 and len(Xte) >= 1


def test_split_requires_at_least_3_rows():
    df = make_frame(2)
    X, y, _, _ = build_features(df)
    with pytest.raises(ValueError):
        split_data(X, y)


def test_safe_r2_single_sample_is_none():
    assert safe_r2(np.array([1.0]), np.array([1.5])) is None


def test_simple_and_multiple_train_and_evaluate():
    df = make_frame(20)
    X, y, _, _ = build_features(df)
    Xtr, _Xv, Xte, ytr, _yv, yte = split_data(X, y)
    simple = train_simple(Xtr, ytr)
    result = evaluate(simple, Xte[["total_sqft"]], yte)
    assert result["mse"] >= 0
    assert result["n"] == len(yte)
    multiple = train_multiple(Xtr, ytr)
    result = evaluate(multiple, Xte, yte)
    assert result["mse"] >= 0
    assert result["r2"] is None or -1e9 < result["r2"] <= 1.0


def test_vif_skipped_when_rows_not_exceeding_features():
    df = make_frame(4)
    X, y, _, _ = build_features(df)
    # With 4 rows, train split has 2 rows < 7 features.
    Xtr, _, _, _, _, _ = split_data(X, y)
    result = vif_table(Xtr)
    assert result["status"] == "skipped"


def test_vif_ok_on_enough_rows():
    df = make_frame(40)
    X, y, _, _ = build_features(df)
    Xtr, _, _, _, _, _ = split_data(X, y)
    result = vif_table(Xtr)
    assert result["status"] == "ok"
    assert len(result["table"]) == Xtr.shape[1]


def test_kmeans_skipped_on_tiny_input():
    df = make_frame(4)
    result = kmeans_segmentation(df)
    assert result["status"] == "skipped"
