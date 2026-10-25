#!/usr/bin/env python3
"""CLI entry point for the House Price Intelligence pipeline.

Usage:
    python run_pipeline.py data/sample_bengaluru.csv
    python run_pipeline.py data/Bengaluru_House_Data.csv --out-dir outputs
    python run_pipeline.py --sample        # bundled sample CSV

Runs all five syllabus phases end to end and writes cleaned CSVs,
dashboard PNGs and metrics.json under the output directory.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from house_price.pipeline import run_pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="House Price Intelligence System pipeline"
    )
    parser.add_argument("csv", nargs="?",
                        help="input housing CSV (or use --sample)")
    parser.add_argument("--sample", action="store_true",
                        help="run on the bundled sample CSV")
    parser.add_argument("--out-dir", default="outputs",
                        help="output directory (default: outputs)")
    args = parser.parse_args(argv)

    if args.sample:
        csv_path = Path(__file__).resolve().parent / "data" / "sample_bengaluru.csv"
    elif args.csv:
        csv_path = Path(args.csv)
    else:
        parser.print_help()
        return 2
    if not csv_path.exists():
        print(f"file not found: {csv_path}", file=sys.stderr)
        return 1

    metrics = run_pipeline(csv_path, args.out_dir)
    print("\n=== Pipeline complete ===")
    print(f"outputs: {args.out_dir}")
    print(f"rows in -> rows after Phase 1: "
          f"{metrics['phase1_clean_report']['rows_in']} -> "
          f"{metrics['phase1_clean_report']['rows_out']}")
    print(f"rows after outlier filter: "
          f"{metrics['outlier_filter']['rows_out']}")
    print(f"split sizes: {metrics['split_sizes']}")
    multi = metrics["multiple_regression"]["test"]
    print(f"multiple regression test MSE: {multi['mse']:.4f}")
    print(f"multiple regression test R²: {multi['r2']}")
    print("\nFull metrics written to "
          f"{Path(args.out_dir) / 'metrics.json'}")
    print(json.dumps(metrics["phase3_stats"]["t_test"], indent=2))
    print(json.dumps(metrics["phase3_stats"]["anova"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
