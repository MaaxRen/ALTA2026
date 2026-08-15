#!/usr/bin/env python3
"""Run POS-based train-to-test experiments on BESSTIE using existing POS summaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from alta2026.pos_experiments import load_pos_dataset, run_all_experiments  # noqa: E402

TRAIN_GOLD_CSV = REPO_ROOT / "data" / "besstie" / "original_train.csv"
TEST_GOLD_CSV = REPO_ROOT / "data" / "besstie" / "test.csv"
TRAIN_POS_SUMMARY_CSV = (
    REPO_ROOT
    / "training_summary"
    / "linguistic_features"
    / "original_train.pos_summary.csv"
)
TEST_POS_SUMMARY_CSV = (
    REPO_ROOT / "training_summary" / "linguistic_features" / "test.pos_summary.csv"
)
OUTPUT_DIR = REPO_ROOT / "training_summary" / "pos_experiments"


def main() -> None:
    train_bundle = load_pos_dataset(TRAIN_GOLD_CSV, TRAIN_POS_SUMMARY_CSV)
    test_bundle = load_pos_dataset(TEST_GOLD_CSV, TEST_POS_SUMMARY_CSV)
    result = run_all_experiments(
        train_bundle=train_bundle,
        test_bundle=test_bundle,
        output_dir=OUTPUT_DIR,
    )
    print(json.dumps(result["summary"], indent=2))
    print()
    print(result["comparison"].to_string(index=False))


if __name__ == "__main__":
    main()
