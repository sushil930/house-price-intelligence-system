"""Integration test: the whole pipeline on the bundled sample CSV."""

import json
from pathlib import Path

from house_price.pipeline import run_pipeline

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_bengaluru.csv"

EXPECTED_CSVS = ["phase1_clean.csv", "phase2_features.csv", "phase3_eda.csv"]
EXPECTED_PNGS = [
    "matplotlib_dashboard.png",
    "seaborn_dashboard.png",
    "pairplot.png",
    "facetgrid.png",
    "residual_plot.png",
]


def test_pipeline_runs_end_to_end_on_sample(tmp_path):
    metrics = run_pipeline(SAMPLE, tmp_path)
    for name in EXPECTED_CSVS:
        path = tmp_path / name
        assert path.exists(), f"missing {name}"
        assert path.stat().st_size > 0
    for name in EXPECTED_PNGS:
        path = tmp_path / name
        assert path.exists(), f"missing {name}"
        assert path.stat().st_size > 0
    metrics_path = tmp_path / "metrics.json"
    assert metrics_path.exists()
    saved = json.loads(metrics_path.read_text())
    assert saved["phase1_clean_report"]["rows_in"] == 6
    assert saved["phase1_clean_report"]["rows_out"] == 5
    assert saved["split_sizes"]["train"] >= 1
    assert saved["split_sizes"]["test"] >= 1
    test_metrics = saved["multiple_regression"]["test"]
    assert test_metrics["mse"] >= 0
    # R-squared is undefined for a single test point on the sample.
    assert test_metrics["r2"] is None
    assert metrics["plots"]["matplotlib_dashboard"].endswith(
        "matplotlib_dashboard.png"
    )
