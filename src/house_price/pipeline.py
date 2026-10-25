"""End-to-end pipeline: raw CSV -> cleaned data -> stats -> plots -> model.

Runs all five syllabus phases in order and writes everything under
``out_dir``:

* ``phase1_clean.csv`` — Phase 1 cleaning pass (Unit I)
* ``phase2_features.csv`` — Phase 2 preprocessing (Unit II)
* ``phase3_eda.csv`` — Phase 3 engineered features (Unit III)
* ``matplotlib_dashboard.png``, ``seaborn_dashboard.png``,
  ``pairplot.png``, ``facetgrid.png``, ``residual_plot.png`` (Unit IV+V)
* ``metrics.json`` — descriptive stats, test results, model metrics
  (MSE / R-squared) and every skipped analysis with its reason.
"""

import json
from pathlib import Path

from . import cleaning, eda_stats, loading, modeling, preprocessing, visualization


def run_pipeline(csv_path, out_dir):
    """Run the full pipeline. Returns the metrics dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"input_csv": str(csv_path)}

    # ---- Phase 1 (Unit I): load + first cleaning pass ----
    raw = loading.load_csv(csv_path)
    metrics["phase1_summary"] = loading.summarize(raw)
    metrics["numpy_price_summary"] = loading.numpy_price_summary(raw)
    metrics["selection_examples"] = loading.select_examples(raw)
    clean1, clean_report = cleaning.phase1_clean(raw)
    metrics["phase1_clean_report"] = clean_report
    clean1.to_csv(out_dir / "phase1_clean.csv", index=False)

    # ---- Phase 2 (Unit II): preprocessing ----
    with_pps = preprocessing.add_price_per_sqft(clean1)
    filtered, outlier_report = preprocessing.apply_outlier_filter(with_pps)
    metrics["outlier_filter"] = outlier_report
    # Sampling demo runs BEFORE one-hot encoding, while the area_type
    # column still exists for stratified sampling.
    metrics["sampling"] = preprocessing.sampling_demo(filtered)
    encoded, encode_report = preprocessing.encode_features(filtered)
    metrics["encoding"] = encode_report
    metrics["measurement_levels"] = preprocessing.MEASUREMENT_LEVELS
    metrics["gaussian_check_price"] = preprocessing.gaussian_check(
        encoded["price"]
    )
    encoded.to_csv(out_dir / "phase2_features.csv", index=False)

    # ---- Phase 3 (Unit III): EDA + inference ----
    eda = eda_stats.engineer_features(encoded)
    metrics["phase3_stats"] = eda_stats.run_all_stats(eda)
    eda.to_csv(out_dir / "phase3_eda.csv", index=False)

    # ---- Phase 4 (Unit IV): dashboards ----
    plots = {}
    plots["matplotlib_dashboard"] = str(
        visualization.matplotlib_dashboard(
            eda, out_dir / "matplotlib_dashboard.png"
        )
    )
    plots["seaborn_dashboard"] = str(
        visualization.seaborn_dashboard(eda, out_dir / "seaborn_dashboard.png")
    )
    plots["pairplot"] = str(
        visualization.pairplot(eda, out_dir / "pairplot.png")
    )
    facet = visualization.facetgrid(eda, out_dir / "facetgrid.png")
    plots["facetgrid"] = str(facet) if facet else None
    metrics["plots"] = plots

    # ---- Phase 5 (Unit V): regression ----
    X, y, features, dropped = modeling.build_features(eda)
    metrics["model_features"] = features
    metrics["model_rows_dropped_missing"] = int(dropped)
    splits = modeling.split_data(X, y)
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    metrics["split_sizes"] = {
        "train": len(X_train),
        "validation": len(X_val),
        "test": len(X_test),
    }

    simple = modeling.train_simple(X_train, y_train)
    metrics["simple_regression"] = {
        "feature": "total_sqft",
        "coefficient": float(simple.coef_[0]),
        "intercept": float(simple.intercept_),
        "validation": modeling.evaluate(
            simple, X_val[["total_sqft"]], y_val
        ),
        "test": modeling.evaluate(simple, X_test[["total_sqft"]], y_test),
    }

    multiple = modeling.train_multiple(X_train, y_train)
    test_pred = multiple.predict(X_test)
    metrics["multiple_regression"] = {
        "coefficients": {
            name: float(coef)
            for name, coef in zip(features, multiple.coef_)
        },
        "intercept": float(multiple.intercept_),
        "validation": modeling.evaluate(multiple, X_val, y_val),
        "test": modeling.evaluate(multiple, X_test, y_test),
    }
    metrics["vif"] = modeling.vif_table(X_train)
    metrics["kmeans_segmentation"] = modeling.kmeans_segmentation(eda)
    metrics["plots"]["residual_plot"] = str(
        modeling.residual_plot(
            y_test, test_pred, out_dir / "residual_plot.png"
        )
    )

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    return metrics
