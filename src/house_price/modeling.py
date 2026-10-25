"""Phase 5 price prediction (Unit V).

Syllabus coverage: supervised vs unsupervised framing, regression vs
classification, train / validation / test splits, simple and multiple
linear regression, regression assumptions (residual plots,
multicollinearity via VIF), and evaluation with MSE and R-squared.

VIF is computed with scikit-learn OLS (regress each feature on the
others) so no statsmodels dependency is needed. Metrics that are
undefined on tiny test sets (R-squared on a single test point) are
reported as ``None`` with a reason, never fabricated.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_NUMERIC_FEATURES = [
    "total_sqft",
    "bath",
    "bhk",
    "balcony",
    "location_tier",
    "room_density",
    "is_premium",
]


def build_features(df):
    """Return (X, y, feature_names) for modelling.

    Numeric base features plus any ``area_type_*`` one-hot columns
    present in the frame; rows missing a feature or the target are
    dropped and counted.
    """
    dummies = [c for c in df.columns if c.startswith("area_type_")]
    features = [c for c in BASE_NUMERIC_FEATURES if c in df.columns] + dummies
    usable = df.dropna(subset=features + ["price"]).copy()
    return usable[features], usable["price"], features, len(df) - len(usable)


def split_data(X, y, test_size=0.2, val_size=0.25, random_state=42):
    """Train / validation / test split.

    First carves out the test set, then splits the remainder into
    train and validation. Requires at least 3 rows; with 5 rows this
    yields 3 train / 1 validation / 1 test.
    """
    if len(X) < 3:
        raise ValueError(
            f"need at least 3 rows to split, got {len(X)}"
        )
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def safe_r2(y_true, y_pred):
    """R-squared, or ``None`` when it is undefined (< 2 samples)."""
    if len(y_true) < 2:
        return None
    try:
        value = float(r2_score(y_true, y_pred))
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def evaluate(model, X, y):
    """MSE and R-squared for a fitted model on a given split."""
    pred = model.predict(X)
    return {
        "n": len(y),
        "mse": float(mean_squared_error(y, pred)),
        "r2": safe_r2(y, pred),
    }


def train_simple(X_train, y_train):
    """Simple linear regression: price ~ total_sqft."""
    model = LinearRegression().fit(X_train[["total_sqft"]], y_train)
    return model


def train_multiple(X_train, y_train):
    """Multiple linear regression on all features."""
    return LinearRegression().fit(X_train, y_train)


def vif_table(X_train):
    """Variance inflation factors via scikit-learn OLS.

    Skipped when there are not more training rows than features,
    because VIF is degenerate in that regime (R-squared saturates).
    """
    n, p = X_train.shape
    if n <= p:
        return {"status": "skipped",
                "reason": f"need more training rows than features "
                          f"(rows={n}, features={p})"}
    rows = []
    X = X_train.to_numpy(dtype=float)
    for i, name in enumerate(X_train.columns):
        others = np.delete(X, i, axis=1)
        target = X[:, i]
        if np.std(target) == 0:
            rows.append({"feature": name, "vif": None,
                         "note": "constant feature"})
            continue
        r2 = LinearRegression().fit(others, target).score(others, target)
        vif = float(1.0 / (1.0 - r2)) if r2 < 1.0 else float("inf")
        rows.append({"feature": name, "vif": vif})
    return {"status": "ok", "table": rows}


def residual_plot(y_true, y_pred, path):
    """Residuals vs predicted values (assumption check)."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.3)
    ax.axhline(0, color="red")
    ax.set_xlabel("Predicted price (lakh)")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual Plot — Multiple Regression")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def kmeans_segmentation(df, n_clusters=3, random_state=42):
    """Optional unsupervised touch: segment listings by sqft and price.

    Skipped on tiny inputs where clusters would be meaningless.
    """
    usable = df.dropna(subset=["total_sqft", "price"])
    if len(usable) < max(10, n_clusters * 2):
        return {"status": "skipped",
                "reason": f"need >= {max(10, n_clusters * 2)} rows, "
                          f"got {len(usable)}"}
    X = StandardScaler().fit_transform(usable[["total_sqft", "price"]])
    labels = KMeans(n_clusters=n_clusters, n_init=10,
                    random_state=random_state).fit_predict(X)
    return {"status": "ok", "n_clusters": n_clusters,
            "cluster_sizes": [int((labels == k).sum())
                              for k in range(n_clusters)]}
