#!/usr/bin/env python3
"""Run a resumable, validation-only hyperparameter search across BESSTIE splits."""

from __future__ import annotations

import gc
import hashlib
import itertools
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from alta2026.multitask_training import (  # noqa: E402
    SharedEncoderTwoHeadClassifier,
    TrainingConfig,
    run_training,
)


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to tune a standard local Hugging Face backbone."""

    local_path: Path
    strategy: Literal["full", "lora"]
    pooling: Literal["cls", "last_token"]
    padding_side: Literal["left", "right"]
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    learning_rates: tuple[float, ...]
    lora_ranks: tuple[int, ...] = (16,)
    lora_alpha_multipliers: tuple[int, ...] = (2,)
    lora_dropouts: tuple[float, ...] = (0.05,)


# Add another standard encoder or decoder here. No search-loop changes are needed.
MODEL_REGISTRY = {
    "qwen": ModelSpec(
        local_path=REPO_ROOT / "resources" / "pretrained_model" / "qwen3-embedding-0.6b",
        strategy="lora",
        pooling="last_token",
        padding_side="left",
        train_batch_size=8,
        eval_batch_size=16,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        learning_rates=(5e-5, 1e-4, 2e-4, 4e-4),
        lora_ranks=(8, 16, 32, 64),
        lora_alpha_multipliers=(1, 2, 4),
        lora_dropouts=(0.0, 0.05, 0.1),
    ),
    "roberta": ModelSpec(
        local_path=REPO_ROOT / "resources" / "pretrained_model" / "roberta",
        strategy="full",
        pooling="cls",
        padding_side="right",
        train_batch_size=16,
        eval_batch_size=32,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        learning_rates=(5e-6, 1e-5, 2e-5, 3e-5, 5e-5),
    ),
}

# Edit constants here; this script intentionally has no command-line arguments.
MODELS_TO_TUNE = ("qwen",)
SEARCH_NAME = "multitask_random_v1"
NUM_CONFIGURATIONS = {"qwen": 48, "roberta": 32}
SEARCH_SEED = 20260806
SPLIT_SEEDS = (2026, 2027, 2028, 2029, 2030)
TRAINING_SEED = 42

DATA_DIR = REPO_ROOT / "data" / "besstie"
TUNING_ROOT = REPO_ROOT / "training_summary" / "hyperparameter_tuning" / SEARCH_NAME
LOCAL_FILES_ONLY = True

MAX_EPOCHS = 8
EARLY_STOPPING_PATIENCE = 2
NUM_WORKERS = 4
MIXED_PRECISION = "bf16"
MAX_GRAD_NORM = 1.0

# Common search space. Effective batch size remains 16 for both registered models.
MAX_LENGTHS = (128, 256, 384)
HEAD_TYPES = ("linear", "mlp")
MLP_HIDDEN_RATIOS = (0.25, 0.5, 0.75)
DROPOUTS = (0.0, 0.05, 0.1, 0.2)
WEIGHT_DECAYS = (0.0, 0.01, 0.05)
WARMUP_RATIOS = (0.0, 0.05, 0.1)
LOSS_WEIGHTINGS = (
    "task_class",
    "task_variety_class",
    "adaptive_task_variety_class",
    "group_dro",
)
ADAPTIVE_BETAS = (0.1, 0.3, 0.5)
ADAPTIVE_TEMPERATURES = (0.1, 0.25, 0.5)
VARIETY_BOUNDS = ((0.25, 0.75), (0.1, 0.9))
GROUP_DRO_STEP_SIZES = (0.001, 0.01, 0.05)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def configuration_id(configuration: dict[str, object]) -> str:
    digest = hashlib.sha256(canonical_json(configuration).encode()).hexdigest()[:12]
    return f"cfg_{digest}"


def default_configuration(model_name: str, spec: ModelSpec) -> dict[str, object]:
    configuration: dict[str, object] = {
        "model": model_name,
        "learning_rate": 2e-4 if spec.strategy == "lora" else 2e-5,
        "max_length": 256,
        "head_type": "linear",
        "mlp_hidden_ratio": None,
        "dropout": 0.1,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "loss_weighting": "adaptive_task_variety_class",
        "adaptive_weight_beta": 0.3,
        "adaptive_weight_temperature": 0.25,
        "min_variety_weight": 0.25,
        "max_variety_weight": 0.75,
        "group_dro_step_size": 0.01,
    }
    if spec.strategy == "lora":
        configuration.update(
            {"lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05}
        )
    return configuration


def sample_configuration(
    model_name: str, spec: ModelSpec, rng: random.Random
) -> dict[str, object]:
    head_type = rng.choice(HEAD_TYPES)
    loss_weighting = rng.choice(LOSS_WEIGHTINGS)
    minimum, maximum = (
        rng.choice(VARIETY_BOUNDS)
        if loss_weighting == "adaptive_task_variety_class"
        else (0.25, 0.75)
    )
    configuration: dict[str, object] = {
        "model": model_name,
        "learning_rate": rng.choice(spec.learning_rates),
        "max_length": rng.choice(MAX_LENGTHS),
        "head_type": head_type,
        "mlp_hidden_ratio": rng.choice(MLP_HIDDEN_RATIOS) if head_type == "mlp" else None,
        "dropout": rng.choice(DROPOUTS),
        "weight_decay": rng.choice(WEIGHT_DECAYS),
        "warmup_ratio": rng.choice(WARMUP_RATIOS),
        "loss_weighting": loss_weighting,
        "adaptive_weight_beta": (
            rng.choice(ADAPTIVE_BETAS)
            if loss_weighting == "adaptive_task_variety_class"
            else 0.3
        ),
        "adaptive_weight_temperature": (
            rng.choice(ADAPTIVE_TEMPERATURES)
            if loss_weighting == "adaptive_task_variety_class"
            else 0.25
        ),
        "min_variety_weight": minimum,
        "max_variety_weight": maximum,
        "group_dro_step_size": (
            rng.choice(GROUP_DRO_STEP_SIZES) if loss_weighting == "group_dro" else 0.01
        ),
    }
    if spec.strategy == "lora":
        rank = rng.choice(spec.lora_ranks)
        configuration.update(
            {
                "lora_r": rank,
                "lora_alpha": rank * rng.choice(spec.lora_alpha_multipliers),
                "lora_dropout": rng.choice(spec.lora_dropouts),
            }
        )
    return configuration


def build_search_manifest(model_name: str, spec: ModelSpec) -> list[dict[str, object]]:
    rng = random.Random(f"{SEARCH_SEED}:{model_name}")
    requested = NUM_CONFIGURATIONS[model_name]
    configurations = [default_configuration(model_name, spec)]
    seen = {canonical_json(configurations[0])}
    while len(configurations) < requested:
        candidate = sample_configuration(model_name, spec, rng)
        key = canonical_json(candidate)
        if key not in seen:
            seen.add(key)
            configurations.append(candidate)
    return [
        {"trial": trial, "config_id": configuration_id(config), "parameters": config}
        for trial, config in enumerate(configurations, start=1)
    ]


def validate_split_files() -> None:
    missing = []
    for split_seed, split_name in itertools.product(
        SPLIT_SEEDS, ("train.csv", "validation.csv")
    ):
        path = DATA_DIR / "splits" / f"seed_{split_seed}" / split_name
        if not path.is_file():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Missing tuning splits. Run notebooks/eda.ipynb before tuning: "
            + ", ".join(missing)
        )


def build_tokenizer(spec: ModelSpec):
    tokenizer = AutoTokenizer.from_pretrained(
        str(spec.local_path),
        use_fast=True,
        padding_side=spec.padding_side,
        local_files_only=LOCAL_FILES_ONLY,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def make_model_factory(
    spec: ModelSpec,
    configuration: dict[str, object],
    hidden_size: int,
):
    mlp_hidden_size = None
    if configuration["head_type"] == "mlp":
        mlp_hidden_size = round(hidden_size * float(configuration["mlp_hidden_ratio"]))

    def model_factory() -> SharedEncoderTwoHeadClassifier:
        backbone = AutoModel.from_pretrained(
            str(spec.local_path), local_files_only=LOCAL_FILES_ONLY
        )
        if spec.strategy == "lora":
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except ImportError as exc:
                raise SystemExit("LoRA tuning requires PEFT.") from exc
            if hasattr(backbone.config, "use_cache"):
                backbone.config.use_cache = False
            backbone = get_peft_model(
                backbone,
                LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=int(configuration["lora_r"]),
                    lora_alpha=int(configuration["lora_alpha"]),
                    lora_dropout=float(configuration["lora_dropout"]),
                    target_modules="all-linear",
                    bias="none",
                ),
            )
        else:
            for parameter in backbone.parameters():
                parameter.requires_grad = True
        return SharedEncoderTwoHeadClassifier(
            backbone=backbone,
            pooling=spec.pooling,
            dropout=float(configuration["dropout"]),
            head_type=str(configuration["head_type"]),
            mlp_hidden_size=mlp_hidden_size,
        )

    return model_factory, mlp_hidden_size


def write_leaderboard(model_root: Path) -> None:
    rows = []
    for aggregate_path in sorted(model_root.glob("cfg_*/aggregate.json")):
        aggregate = json.loads(aggregate_path.read_text())
        rows.append(
            {
                "config_id": aggregate["config_id"],
                "trial": aggregate["trial"],
                "mean_validation_score": aggregate["mean_validation_score"],
                "std_validation_score": aggregate["std_validation_score"],
                "min_validation_score": aggregate["min_validation_score"],
                "mean_best_epoch": aggregate["mean_best_epoch"],
                **aggregate["parameters"],
            }
        )
    if not rows:
        return
    leaderboard = pd.DataFrame(rows).sort_values(
        ["mean_validation_score", "std_validation_score"],
        ascending=[False, True],
    )
    leaderboard.to_csv(model_root / "leaderboard.csv", index=False)
    (model_root / "leaderboard.json").write_text(
        json.dumps(leaderboard.to_dict(orient="records"), indent=2) + "\n"
    )


def tune_model(model_name: str) -> None:
    spec = MODEL_REGISTRY[model_name]
    if not spec.local_path.is_dir():
        raise FileNotFoundError(f"Local model directory not found: {spec.local_path}")

    model_root = TUNING_ROOT / model_name
    model_root.mkdir(parents=True, exist_ok=True)
    manifest_path = model_root / "search_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        print(f"Resuming the existing manifest at {manifest_path}")
    else:
        manifest = build_search_manifest(model_name, spec)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    tokenizer = build_tokenizer(spec)
    model_config = AutoConfig.from_pretrained(
        str(spec.local_path), local_files_only=LOCAL_FILES_ONLY
    )
    hidden_size = int(model_config.hidden_size)

    for manifest_entry in manifest:
        trial = int(manifest_entry["trial"])
        config_id = str(manifest_entry["config_id"])
        parameters = dict(manifest_entry["parameters"])
        config_root = model_root / config_id
        config_root.mkdir(parents=True, exist_ok=True)
        (config_root / "configuration.json").write_text(
            json.dumps(manifest_entry, indent=2) + "\n"
        )

        seed_results = {}
        print(f"\n=== {model_name} trial {trial}/{len(manifest)}: {config_id} ===")
        for split_seed in SPLIT_SEEDS:
            seed_summary_dir = config_root / f"seed_{split_seed}"
            result_path = seed_summary_dir / "validation_result.json"
            if result_path.is_file():
                validation_result = json.loads(result_path.read_text())
                print(f"Skipping completed {config_id}, split {split_seed}")
            else:
                model_factory, mlp_hidden_size = make_model_factory(
                    spec, parameters, hidden_size
                )
                training_config = TrainingConfig(
                    model_name=str(spec.local_path),
                    output_dir=str(
                        REPO_ROOT
                        / "model_checkpoints"
                        / "tuning"
                        / SEARCH_NAME
                        / model_name
                        / config_id
                        / f"seed_{split_seed}"
                    ),
                    summary_dir=str(seed_summary_dir),
                    pooling=spec.pooling,
                    data_dir=str(DATA_DIR),
                    split_seed=split_seed,
                    epochs=MAX_EPOCHS,
                    train_batch_size=spec.train_batch_size,
                    eval_batch_size=spec.eval_batch_size,
                    gradient_accumulation_steps=spec.gradient_accumulation_steps,
                    learning_rate=float(parameters["learning_rate"]),
                    weight_decay=float(parameters["weight_decay"]),
                    warmup_ratio=float(parameters["warmup_ratio"]),
                    max_length=int(parameters["max_length"]),
                    dropout=float(parameters["dropout"]),
                    max_grad_norm=MAX_GRAD_NORM,
                    seed=TRAINING_SEED,
                    num_workers=NUM_WORKERS,
                    mixed_precision=MIXED_PRECISION,
                    gradient_checkpointing=spec.gradient_checkpointing,
                    loss_weighting=str(parameters["loss_weighting"]),
                    selection_metric="min_variety_mean_macro_f1",
                    adaptive_weight_beta=float(parameters["adaptive_weight_beta"]),
                    adaptive_weight_temperature=float(
                        parameters["adaptive_weight_temperature"]
                    ),
                    min_variety_weight=float(parameters["min_variety_weight"]),
                    max_variety_weight=float(parameters["max_variety_weight"]),
                    head_type=str(parameters["head_type"]),
                    mlp_hidden_size=mlp_hidden_size,
                    evaluate_test=False,
                    save_model_checkpoint=False,
                    early_stopping_patience=EARLY_STOPPING_PATIENCE,
                    group_dro_step_size=float(parameters["group_dro_step_size"]),
                )
                method = f"tuning_{spec.strategy}_{parameters['loss_weighting']}"
                result = run_training(
                    training_config, tokenizer, model_factory, method=method
                )
                validation_result = {
                    "selected_epoch": result["best_epoch"],
                    "best_validation_score": result["best_validation_score"],
                    "validation_metrics": result["validation_metrics"],
                }

            seed_results[str(split_seed)] = validation_result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

        scores = np.asarray(
            [result["best_validation_score"] for result in seed_results.values()],
            dtype=float,
        )
        epochs = np.asarray(
            [result["selected_epoch"] for result in seed_results.values()], dtype=float
        )
        aggregate = {
            "model": model_name,
            "config_id": config_id,
            "trial": trial,
            "parameters": parameters,
            "split_seeds": list(SPLIT_SEEDS),
            "mean_validation_score": float(scores.mean()),
            "std_validation_score": float(scores.std(ddof=1)),
            "min_validation_score": float(scores.min()),
            "max_validation_score": float(scores.max()),
            "mean_best_epoch": float(epochs.mean()),
            "seed_results": seed_results,
            "test_evaluated": False,
        }
        (config_root / "aggregate.json").write_text(
            json.dumps(aggregate, indent=2) + "\n"
        )
        write_leaderboard(model_root)
        print(
            f"{config_id}: validation score "
            f"{aggregate['mean_validation_score']:.4f} ± "
            f"{aggregate['std_validation_score']:.4f}"
        )


def main() -> None:
    validate_split_files()
    unknown = set(MODELS_TO_TUNE) - set(MODEL_REGISTRY)
    if unknown:
        raise ValueError(f"Unknown models in MODELS_TO_TUNE: {sorted(unknown)}")
    TUNING_ROOT.mkdir(parents=True, exist_ok=True)
    (TUNING_ROOT / "search_settings.json").write_text(
        json.dumps(
            {
                "models": list(MODELS_TO_TUNE),
                "search_name": SEARCH_NAME,
                "split_seeds": list(SPLIT_SEEDS),
                "training_seed": TRAINING_SEED,
                "max_epochs": MAX_EPOCHS,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "model_registry": {
                    name: {**asdict(spec), "local_path": str(spec.local_path)}
                    for name, spec in MODEL_REGISTRY.items()
                },
                "test_evaluated": False,
            },
            indent=2,
        )
        + "\n"
    )
    for model_name in MODELS_TO_TUNE:
        tune_model(model_name)


if __name__ == "__main__":
    main()
