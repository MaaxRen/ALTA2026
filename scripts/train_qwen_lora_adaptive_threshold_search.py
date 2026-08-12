#!/usr/bin/env python3
"""Qwen 0.6B adaptive weighting with validation-tuned sarcasm thresholds."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from alta2026.qwen_lora_experiment import (  # noqa: E402
    QwenLoraExperimentSettings,
    run_qwen_lora_experiment,
)

# Edit this configuration block before submitting the HPC job.
SETTINGS = QwenLoraExperimentSettings(
    model_name=REPO_ROOT / "resources" / "pretrained_model" / "qwen3-embedding-0.6b",
    data_dir=REPO_ROOT / "data" / "besstie",
    output_dir=REPO_ROOT / "model_checkpoints" / "qwen_embedding_0.6b_lora_adaptive_threshold_search",
    summary_dir=REPO_ROOT / "training_summary" / "runs" / "qwen_embedding_0.6b_lora_adaptive_threshold_search",
    split_seeds=(2026, 2027, 2028, 2029, 2030),
    method="qwen_0.6b_lora_adaptive_threshold_search",
    loss_weighting="adaptive_task_variety_class",
    selection_metric="min_variety_mean_macro_f1",
    epochs=5,
    train_batch_size=8,
    eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    max_length=256,
    dropout=0.1,
    mixed_precision="bf16",
    gradient_checkpointing=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    attention_implementation="sdpa",
    adaptive_weight_beta=0.3,
    adaptive_weight_temperature=0.25,
    min_variety_weight=0.25,
    max_variety_weight=0.75,
    search_sarcasm_thresholds=True,
    sarcasm_threshold_min=0.01,
    sarcasm_threshold_max=0.99,
    sarcasm_threshold_steps=99,
)


if __name__ == "__main__":
    run_qwen_lora_experiment(SETTINGS)
