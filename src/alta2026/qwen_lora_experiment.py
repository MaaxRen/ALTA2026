"""Reusable runner for locally staged Qwen embedding LoRA experiments."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer

from alta2026.multitask_training import (
    SharedEncoderTwoHeadClassifier,
    TrainingConfig,
    run_training,
)


@dataclass(frozen=True)
class QwenLoraExperimentSettings:
    model_name: Path
    data_dir: Path
    output_dir: Path
    summary_dir: Path
    split_seeds: tuple[int, ...]
    method: str
    loss_weighting: Literal[
        "task_class", "adaptive_task_variety_class", "group_dro"
    ]
    selection_metric: Literal[
        "overall_mean_macro_f1", "min_variety_mean_macro_f1"
    ]
    epochs: int = 5
    train_batch_size: int = 8
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 256
    dropout: float = 0.1
    max_grad_norm: float = 1.0
    training_seed: int = 42
    num_workers: int = 4
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    gradient_checkpointing: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    attention_implementation: str = "sdpa"
    adaptive_weight_beta: float = 0.3
    adaptive_weight_temperature: float = 0.25
    min_variety_weight: float = 0.25
    max_variety_weight: float = 0.75
    group_dro_step_size: float = 0.01
    search_sarcasm_thresholds: bool = False
    sarcasm_threshold_min: float = 0.01
    sarcasm_threshold_max: float = 0.99
    sarcasm_threshold_steps: int = 99
    train_limit: int | None = None
    validation_limit: int | None = None
    test_limit: int | None = None
    local_files_only: bool = True


def _load_dtype(mixed_precision: str) -> torch.dtype:
    """Use reduced precision on CUDA and portable FP32 elsewhere."""
    if not torch.cuda.is_available():
        return torch.float32
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return torch.float32


def run_qwen_lora_experiment(settings: QwenLoraExperimentSettings) -> None:
    """Run every configured split and write aggregate JSON and CSV summaries."""
    if not settings.model_name.is_dir():
        raise FileNotFoundError(
            f"Local model directory not found: {settings.model_name}. "
            "Stage the Qwen model before copying the project to the HPC."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(settings.model_name),
        use_fast=True,
        padding_side="left",
        local_files_only=settings.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {}
    for split_seed in settings.split_seeds:
        print(f"\n=== {settings.method}, split seed {split_seed} ===")
        config = TrainingConfig(
            model_name=str(settings.model_name),
            output_dir=str(settings.output_dir / f"seed_{split_seed}"),
            summary_dir=str(settings.summary_dir / f"seed_{split_seed}"),
            pooling="last_token",
            data_dir=str(settings.data_dir),
            split_seed=split_seed,
            epochs=settings.epochs,
            train_batch_size=settings.train_batch_size,
            eval_batch_size=settings.eval_batch_size,
            gradient_accumulation_steps=settings.gradient_accumulation_steps,
            learning_rate=settings.learning_rate,
            weight_decay=settings.weight_decay,
            warmup_ratio=settings.warmup_ratio,
            max_length=settings.max_length,
            dropout=settings.dropout,
            max_grad_norm=settings.max_grad_norm,
            seed=settings.training_seed,
            num_workers=settings.num_workers,
            mixed_precision=settings.mixed_precision,
            gradient_checkpointing=settings.gradient_checkpointing,
            train_limit=settings.train_limit,
            validation_limit=settings.validation_limit,
            test_limit=settings.test_limit,
            loss_weighting=settings.loss_weighting,
            selection_metric=settings.selection_metric,
            adaptive_weight_beta=settings.adaptive_weight_beta,
            adaptive_weight_temperature=settings.adaptive_weight_temperature,
            min_variety_weight=settings.min_variety_weight,
            max_variety_weight=settings.max_variety_weight,
            group_dro_step_size=settings.group_dro_step_size,
            search_sarcasm_thresholds=settings.search_sarcasm_thresholds,
            sarcasm_threshold_min=settings.sarcasm_threshold_min,
            sarcasm_threshold_max=settings.sarcasm_threshold_max,
            sarcasm_threshold_steps=settings.sarcasm_threshold_steps,
            head_type="linear",
        )

        def model_factory() -> SharedEncoderTwoHeadClassifier:
            backbone = AutoModel.from_pretrained(
                config.model_name,
                local_files_only=settings.local_files_only,
                torch_dtype=_load_dtype(settings.mixed_precision),
                attn_implementation=settings.attention_implementation,
            )
            if hasattr(backbone.config, "use_cache"):
                backbone.config.use_cache = False
            backbone = get_peft_model(
                backbone,
                LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=settings.lora_r,
                    lora_alpha=settings.lora_alpha,
                    lora_dropout=settings.lora_dropout,
                    target_modules="all-linear",
                    bias="none",
                ),
            )
            return SharedEncoderTwoHeadClassifier(
                backbone=backbone,
                pooling=config.pooling,
                dropout=config.dropout,
                head_type="linear",
            )

        all_results[str(split_seed)] = run_training(
            config, tokenizer, model_factory, method=settings.method
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    settings.summary_dir.mkdir(parents=True, exist_ok=True)
    (settings.summary_dir / "all_results.json").write_text(
        json.dumps(all_results, indent=2) + "\n"
    )
    summary = pd.DataFrame(
        [
            {
                "split_seed": int(split_seed),
                "best_epoch": result["best_epoch"],
                "best_validation_score": result["best_validation_score"],
                **(
                    {
                        f"validation_sarcasm_threshold_{subset}": threshold
                        for subset, threshold in result[
                            "selected_sarcasm_thresholds"
                        ].items()
                    }
                    if result["selected_sarcasm_thresholds"] is not None
                    else {}
                ),
                **{
                    f"test_{key}": value
                    for key, value in result["test_metrics"].items()
                },
            }
            for split_seed, result in all_results.items()
        ]
    )
    summary.to_csv(settings.summary_dir / "all_results.csv", index=False)
    print("\n=== Results across splits ===")
    print(summary.to_string(index=False))
