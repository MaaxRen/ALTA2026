"""Runner shared by the RoBERTa and Qwen Group DRO entry points."""

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

LOCAL_FILES_ONLY = True
DATA_DIR = REPO_ROOT / "data" / "besstie"
SPLIT_SEEDS = (2026, 2027, 2028, 2029, 2030)

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
GROUP_DRO_STEP_SIZE = 0.01

TRAIN_LIMIT = None
VALIDATION_LIMIT = None
TEST_LIMIT = None

MODEL_SETTINGS = {
    "roberta": {
        "model_name": REPO_ROOT / "resources" / "pretrained_model" / "roberta",
        "run_name": "roberta_group_dro",
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
        "run_name": "qwen_embedding_0.6b_lora_group_dro",
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


def run_group_dro_training(model_family: Literal["roberta", "qwen"]) -> None:
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
        print(f"\n=== {model_family} Group DRO, split {split_seed} ===")
        config = TrainingConfig(
            model_name=model_name,
            output_dir=str(output_root / f"seed_{split_seed}"),
            summary_dir=str(summary_root / f"seed_{split_seed}"),
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
            loss_weighting="group_dro",
            selection_metric="min_variety_mean_macro_f1",
            head_type=HEAD_TYPE,
            group_dro_step_size=GROUP_DRO_STEP_SIZE,
        )

        def model_factory() -> SharedEncoderTwoHeadClassifier:
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
            return SharedEncoderTwoHeadClassifier(
                backbone=backbone,
                pooling=config.pooling,
                dropout=config.dropout,
                head_type=config.head_type,
                mlp_hidden_size=config.mlp_hidden_size,
            )

        method = (
            "lora_all_linear_group_dro"
            if settings["strategy"] == "lora"
            else "full_finetuning_group_dro"
        )
        all_results[str(split_seed)] = run_training(
            config, tokenizer, model_factory, method=method
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    summary_root.mkdir(parents=True, exist_ok=True)
    (summary_root / "all_results.json").write_text(
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
    summary.to_csv(summary_root / "all_results.csv", index=False)
    print(summary.to_string(index=False))
