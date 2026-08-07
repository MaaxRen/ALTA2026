"""Runner shared by the separate sentiment and sarcasm model entry points."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from alta2026.multitask_training import TrainingConfig, load_local_besstie  # noqa: E402
from alta2026.single_task_training import (  # noqa: E402
    SingleTaskClassifier,
    train_single_task,
)

LOCAL_FILES_ONLY = True
DATA_DIR = REPO_ROOT / "data" / "besstie"
SPLIT_SEEDS = (2026, 2027, 2028, 2029, 2030)
TASKS = ("sentiment", "sarcasm")

EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_LENGTH = 256
DROPOUT = 0.1
HEAD_TYPE = "linear"
MAX_GRAD_NORM = 1.0
TRAINING_SEED = 42
NUM_WORKERS = 4
MIXED_PRECISION = "bf16"

TRAIN_LIMIT = None
VALIDATION_LIMIT = None
TEST_LIMIT = None

MODEL_SETTINGS = {
    "roberta": {
        "model_name": REPO_ROOT / "resources" / "pretrained_model" / "roberta",
        "run_name": "roberta_separate_tasks",
        "strategy": "full",
        "pooling": "cls",
        "padding_side": "right",
        "train_batch_size": 16,
        "eval_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-5,
        "gradient_checkpointing": False,
    },
    "qwen": {
        "model_name": REPO_ROOT / "resources" / "pretrained_model" / "qwen3-embedding-0.6b",
        "run_name": "qwen_embedding_0.6b_lora_separate_tasks",
        "strategy": "lora",
        "pooling": "last_token",
        "padding_side": "left",
        "train_batch_size": 8,
        "eval_batch_size": 16,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-4,
        "gradient_checkpointing": True,
    },
}

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


def combined_task_metrics(task_results: dict[str, dict]) -> dict[str, float]:
    metrics = {}
    for task, result in task_results.items():
        for key, value in result["test_metrics"].items():
            if key == "selection_score":
                continue
            if key.startswith("overall_"):
                renamed = f"overall_{task}_{key.removeprefix('overall_')}"
            elif key.startswith("en_AU_"):
                renamed = f"en_AU_{task}_{key.removeprefix('en_AU_')}"
            elif key.startswith("en_UK_"):
                renamed = f"en_UK_{task}_{key.removeprefix('en_UK_')}"
            elif key.startswith("google_"):
                renamed = f"google_{task}_{key.removeprefix('google_')}"
            elif key.startswith("reddit_"):
                renamed = f"reddit_{task}_{key.removeprefix('reddit_')}"
            else:
                renamed = f"{task}_{key}"
            metrics[renamed] = value
    metrics["selection_score"] = 0.5 * (
        task_results["sentiment"]["test_metrics"]["selection_score"]
        + task_results["sarcasm"]["test_metrics"]["selection_score"]
    )
    return metrics


def run_separate_task_training(model_family: Literal["roberta", "qwen"]) -> None:
    settings = MODEL_SETTINGS[model_family]
    model_name = str(settings["model_name"])
    output_root = REPO_ROOT / "model_checkpoints" / settings["run_name"]
    summary_root = REPO_ROOT / "training_summary" / "runs" / settings["run_name"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        padding_side=settings["padding_side"],
        local_files_only=LOCAL_FILES_ONLY,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {}
    for split_seed in SPLIT_SEEDS:
        print(f"\n=== {model_family} separate models, split {split_seed} ===")
        datasets = load_local_besstie(
            tokenizer=tokenizer,
            data_dir=DATA_DIR,
            split_seed=split_seed,
            max_length=MAX_LENGTH,
            seed=TRAINING_SEED,
            train_limit=TRAIN_LIMIT,
            validation_limit=VALIDATION_LIMIT,
            test_limit=TEST_LIMIT,
        )
        task_results = {}
        for task in TASKS:
            config = TrainingConfig(
                model_name=model_name,
                output_dir=str(output_root / f"seed_{split_seed}" / task),
                summary_dir=str(summary_root / f"seed_{split_seed}" / task),
                pooling=settings["pooling"],
                data_dir=str(DATA_DIR),
                split_seed=split_seed,
                epochs=EPOCHS,
                train_batch_size=settings["train_batch_size"],
                eval_batch_size=settings["eval_batch_size"],
                gradient_accumulation_steps=settings["gradient_accumulation_steps"],
                learning_rate=settings["learning_rate"],
                weight_decay=WEIGHT_DECAY,
                warmup_ratio=WARMUP_RATIO,
                max_length=MAX_LENGTH,
                dropout=DROPOUT,
                max_grad_norm=MAX_GRAD_NORM,
                seed=TRAINING_SEED,
                num_workers=NUM_WORKERS,
                mixed_precision=MIXED_PRECISION,
                gradient_checkpointing=settings["gradient_checkpointing"],
                train_limit=TRAIN_LIMIT,
                validation_limit=VALIDATION_LIMIT,
                test_limit=TEST_LIMIT,
                loss_weighting="task_variety_class",
                selection_metric="min_variety_mean_macro_f1",
                head_type=HEAD_TYPE,
            )

            def model_factory() -> SingleTaskClassifier:
                backbone = AutoModel.from_pretrained(
                    model_name, local_files_only=LOCAL_FILES_ONLY
                )
                if settings["strategy"] == "lora":
                    try:
                        from peft import LoraConfig, TaskType, get_peft_model
                    except ImportError as exc:
                        raise SystemExit("Qwen LoRA training requires PEFT.") from exc
                    if hasattr(backbone.config, "use_cache"):
                        backbone.config.use_cache = False
                    backbone = get_peft_model(
                        backbone,
                        LoraConfig(
                            task_type=TaskType.FEATURE_EXTRACTION,
                            r=LORA_R,
                            lora_alpha=LORA_ALPHA,
                            lora_dropout=LORA_DROPOUT,
                            target_modules="all-linear",
                            bias="none",
                        ),
                    )
                else:
                    for parameter in backbone.parameters():
                        parameter.requires_grad = True
                return SingleTaskClassifier(
                    backbone=backbone,
                    pooling=config.pooling,
                    dropout=config.dropout,
                    head_type=config.head_type,
                    mlp_hidden_size=config.mlp_hidden_size,
                )

            task_results[task] = train_single_task(
                model_factory(),
                tokenizer,
                datasets,
                config,
                task=task,
                method=f"separate_{settings['strategy']}_task_variety_balanced",
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

        combined = {
            "split_seed": split_seed,
            "best_epochs": {
                task: task_results[task]["best_epoch"] for task in TASKS
            },
            "best_validation_score": 0.5
            * sum(task_results[task]["best_validation_score"] for task in TASKS),
            "test_metrics": combined_task_metrics(task_results),
            "task_results": task_results,
        }
        all_results[str(split_seed)] = combined
        seed_summary = summary_root / f"seed_{split_seed}"
        (seed_summary / "combined_results.json").write_text(
            json.dumps(combined, indent=2) + "\n"
        )

    summary_root.mkdir(parents=True, exist_ok=True)
    (summary_root / "all_results.json").write_text(
        json.dumps(all_results, indent=2) + "\n"
    )
    rows = []
    for split_seed, result in all_results.items():
        rows.append(
            {
                "split_seed": int(split_seed),
                "sentiment_best_epoch": result["best_epochs"]["sentiment"],
                "sarcasm_best_epoch": result["best_epochs"]["sarcasm"],
                "best_validation_score": result["best_validation_score"],
                **{f"test_{key}": value for key, value in result["test_metrics"].items()},
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(summary_root / "all_results.csv", index=False)
    print(summary.to_string(index=False))
