"""Configuration and runner shared by the adaptive minimum-variety entry points."""

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

from alta2026.multitask_training import (  # noqa: E402
    SharedEncoderTwoHeadClassifier,
    TrainingConfig,
    run_training,
)

# Edit this configuration block before submitting an HPC job.
LOCAL_FILES_ONLY = True
DATA_DIR = REPO_ROOT / "data" / "besstie"
SPLIT_SEEDS = (2026, 2027, 2028, 2029, 2030)

EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_LENGTH = 256
DROPOUT = 0.1
MAX_GRAD_NORM = 1.0
TRAINING_SEED = 42
NUM_WORKERS = 4
MIXED_PRECISION = "bf16"

# Validation-driven weighting settings.
ADAPTIVE_WEIGHT_BETA = 0.3
ADAPTIVE_WEIGHT_TEMPERATURE = 0.25
MIN_VARIETY_WEIGHT = 0.25
MAX_VARIETY_WEIGHT = 0.75

# Optional limits for quick smoke tests. Keep as None for complete runs.
TRAIN_LIMIT = None
VALIDATION_LIMIT = None
TEST_LIMIT = None

MODEL_SETTINGS = {
    "roberta": {
        "model_name": REPO_ROOT / "resources" / "pretrained_model" / "roberta",
        "output_dir": REPO_ROOT / "model_checkpoints" / "roberta_adaptive_minvariety",
        "summary_dir": REPO_ROOT / "training_summary" / "runs" / "roberta_adaptive_minvariety",
        "pooling": "cls",
        "train_batch_size": 16,
        "eval_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-5,
        "gradient_checkpointing": False,
    },
    "qwen": {
        "model_name": REPO_ROOT / "resources" / "pretrained_model" / "qwen3-embedding-0.6b",
        "output_dir": REPO_ROOT / "model_checkpoints" / "qwen_embedding_0.6b_lora_adaptive_minvariety",
        "summary_dir": REPO_ROOT / "training_summary" / "runs" / "qwen_embedding_0.6b_lora_adaptive_minvariety",
        "pooling": "last_token",
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


def run_adaptive_training(
    model_family: Literal["roberta", "qwen"],
    head_type: Literal["linear", "mlp"] = "linear",
) -> None:
    settings = MODEL_SETTINGS[model_family]
    run_name = settings["output_dir"].name
    if head_type == "mlp":
        run_name += "_mlp"
    output_dir = REPO_ROOT / "model_checkpoints" / run_name
    summary_dir = REPO_ROOT / "training_summary" / "runs" / run_name
    tokenizer_kwargs = {
        "use_fast": True,
        "local_files_only": LOCAL_FILES_ONLY,
    }
    if model_family == "qwen":
        tokenizer_kwargs["padding_side"] = "left"
    tokenizer = AutoTokenizer.from_pretrained(
        str(settings["model_name"]), **tokenizer_kwargs
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {}
    for split_seed in SPLIT_SEEDS:
        print(f"\n=== {model_family} adaptive split seed {split_seed} ===")
        config = TrainingConfig(
            model_name=str(settings["model_name"]),
            output_dir=str(output_dir / f"seed_{split_seed}"),
            summary_dir=str(summary_dir / f"seed_{split_seed}"),
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
            loss_weighting="adaptive_task_variety_class",
            selection_metric="min_variety_mean_macro_f1",
            adaptive_weight_beta=ADAPTIVE_WEIGHT_BETA,
            adaptive_weight_temperature=ADAPTIVE_WEIGHT_TEMPERATURE,
            min_variety_weight=MIN_VARIETY_WEIGHT,
            max_variety_weight=MAX_VARIETY_WEIGHT,
            head_type=head_type,
        )

        def model_factory() -> SharedEncoderTwoHeadClassifier:
            backbone = AutoModel.from_pretrained(
                config.model_name, local_files_only=LOCAL_FILES_ONLY
            )
            if model_family == "qwen":
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
            return SharedEncoderTwoHeadClassifier(
                backbone=backbone,
                pooling=config.pooling,
                dropout=config.dropout,
                head_type=config.head_type,
                mlp_hidden_size=config.mlp_hidden_size,
            )

        method = (
            "lora_all_linear_adaptive_minvariety"
            if model_family == "qwen"
            else "full_finetuning_adaptive_minvariety"
        )
        if head_type == "mlp":
            method += "_mlp_heads"
        all_results[str(split_seed)] = run_training(
            config, tokenizer, model_factory, method=method
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "all_results.json").write_text(
        json.dumps(all_results, indent=2) + "\n"
    )
    summary = pd.DataFrame(
        [
            {
                "split_seed": int(split_seed),
                "best_epoch": result["best_epoch"],
                "best_validation_score": result["best_validation_score"],
                **{f"test_{key}": value for key, value in result["test_metrics"].items()},
            }
            for split_seed, result in all_results.items()
        ]
    )
    summary.to_csv(summary_dir / "all_results.csv", index=False)
    print(f"\n=== {model_family} adaptive results across splits ===")
    print(summary.to_string(index=False))
