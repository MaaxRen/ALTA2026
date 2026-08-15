#!/usr/bin/env python3
"""Generate Stanford CoreNLP POS-tagging artifacts for BESSTIE CSV files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from alta2026.pos_features import process_csv_to_pos_artifacts  # noqa: E402


DEFAULT_INPUTS = (
    REPO_ROOT / "data" / "besstie" / "all.csv",
    REPO_ROOT / "data" / "besstie" / "original_train.csv",
    REPO_ROOT / "data" / "besstie" / "test.csv",
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "training_summary" / "linguistic_features"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        action="append",
        dest="input_csvs",
        help="CSV to POS-tag. Repeatable. Defaults to all/main BESSTIE CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for derived POS artifacts.",
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:9000",
        help="Stanford CoreNLP server URL.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP timeout for each CoreNLP request.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csvs = (
        [Path(path) for path in args.input_csvs]
        if args.input_csvs
        else list(DEFAULT_INPUTS)
    )
    results = [
        process_csv_to_pos_artifacts(
            path,
            args.output_dir,
            server_url=args.server_url,
            timeout_seconds=args.timeout_seconds,
            limit=args.limit,
        )
        for path in input_csvs
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
