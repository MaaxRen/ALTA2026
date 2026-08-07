#!/usr/bin/env python3
"""Fully fine-tune RoBERTa with shared sentiment and sarcasm heads on BESSTIE."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from alta2026.multitask_training import (  # noqa: E402
    SharedEncoderTwoHeadClassifier,
    TrainingConfig,
    run_training,
)

# Edit this configuration block before submitting the HPC job.
MODEL_NAME = str(REPO_ROOT / "resources" / "pretrained_model" / "roberta")
LOCAL_FILES_ONLY = True
DATA_DIR = REPO_ROOT / "data" / "besstie"
OUTPUT_DIR = REPO_ROOT / "model_checkpoints" / "roberta_multitask"
TRAINING_SUMMARY_DIR = REPO_ROOT / "training_summary" / "runs" / "roberta_multitask"
SPLIT_SEEDS = (2026, 2027, 2028, 2029, 2030)

EPOCHS = 5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_LENGTH = 256
DROPOUT = 0.1
MAX_GRAD_NORM = 1.0
TRAINING_SEED = 42
NUM_WORKERS = 4
MIXED_PRECISION = "bf16"
GRADIENT_CHECKPOINTING = False

# Optional limits for quick smoke tests. Keep as None for complete runs.
TRAIN_LIMIT = None
VALIDATION_LIMIT = None
TEST_LIMIT = None


def main() -> None:
    output_root = OUTPUT_DIR
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        local_files_only=LOCAL_FILES_ONLY,
    )
    all_results = {}

    for split_seed in SPLIT_SEEDS:
        print(f"\n=== RoBERTa split seed {split_seed} ===")
        config = TrainingConfig(
            model_name=MODEL_NAME,
            output_dir=str(output_root / f"seed_{split_seed}"),
            summary_dir=str(TRAINING_SUMMARY_DIR / f"seed_{split_seed}"),
            pooling="cls",
            data_dir=str(DATA_DIR),
            split_seed=split_seed,
            epochs=EPOCHS,
            train_batch_size=TRAIN_BATCH_SIZE,
            eval_batch_size=EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,
            max_length=MAX_LENGTH,
            dropout=DROPOUT,
            max_grad_norm=MAX_GRAD_NORM,
            seed=TRAINING_SEED,
            num_workers=NUM_WORKERS,
            mixed_precision=MIXED_PRECISION,
            gradient_checkpointing=GRADIENT_CHECKPOINTING,
            train_limit=TRAIN_LIMIT,
            validation_limit=VALIDATION_LIMIT,
            test_limit=TEST_LIMIT,
        )

        def model_factory() -> SharedEncoderTwoHeadClassifier:
            backbone = AutoModel.from_pretrained(
                config.model_name,
                local_files_only=LOCAL_FILES_ONLY,
            )
            for parameter in backbone.parameters():
                parameter.requires_grad = True
            return SharedEncoderTwoHeadClassifier(
                backbone=backbone,
                pooling=config.pooling,
                dropout=config.dropout,
            )

        all_results[str(split_seed)] = run_training(
            config, tokenizer, model_factory, method="full_finetuning"
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    TRAINING_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    (TRAINING_SUMMARY_DIR / "all_results.json").write_text(
        json.dumps(all_results, indent=2) + "\n"
    )
    summary = pd.DataFrame(
        [
            {
                "split_seed": int(split_seed),
                "best_epoch": result["best_epoch"],
                "best_validation_mean_macro_f1": result[
                    "best_validation_mean_macro_f1"
                ],
                **{f"test_{key}": value for key, value in result["test_metrics"].items()},
            }
            for split_seed, result in all_results.items()
        ]
    )
    summary.to_csv(TRAINING_SUMMARY_DIR / "all_results.csv", index=False)
    print("\n=== RoBERTa results across splits ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
