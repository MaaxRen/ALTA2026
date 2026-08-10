"""Shared multitask training utilities for the BESSTIE baselines."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup

SUBSETS = ("en_AU", "en_UK")
SUBSET_TO_ID = {name: index for index, name in enumerate(SUBSETS)}
ID_TO_SUBSET = {index: name for name, index in SUBSET_TO_ID.items()}
SOURCE_TO_ID = {"google": 0, "reddit": 1}
ID_TO_SOURCE = {index: name for name, index in SOURCE_TO_ID.items()}


@dataclass
class TrainingConfig:
    model_name: str
    output_dir: str
    summary_dir: str
    pooling: Literal["cls", "last_token"]
    data_dir: str = "data/besstie"
    split_seed: int = 2026
    epochs: int = 3
    train_batch_size: int = 8
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 256
    dropout: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 42
    num_workers: int = 0
    mixed_precision: Literal["no", "fp16", "bf16"] = "no"
    gradient_checkpointing: bool = False
    train_limit: int | None = None
    validation_limit: int | None = None
    test_limit: int | None = None
    loss_weighting: Literal[
        "task_class",
        "task_variety_class",
        "adaptive_task_variety_class",
        "group_dro",
    ] = "task_class"
    selection_metric: Literal[
        "overall_mean_macro_f1", "min_variety_mean_macro_f1"
    ] = "overall_mean_macro_f1"
    adaptive_weight_beta: float = 0.3
    adaptive_weight_temperature: float = 0.25
    min_variety_weight: float = 0.25
    max_variety_weight: float = 0.75
    head_type: Literal["linear", "mlp"] = "linear"
    mlp_hidden_size: int | None = None
    evaluate_test: bool = True
    save_model_checkpoint: bool = True
    early_stopping_patience: int | None = None
    group_dro_step_size: float = 0.01
    search_sarcasm_thresholds: bool = False
    sarcasm_threshold_min: float = 0.01
    sarcasm_threshold_max: float = 0.99
    sarcasm_threshold_steps: int = 99


class MultitaskDataCollator:
    """Dynamically pad text features and retain numeric labels/metadata."""

    metadata_keys = (
        "sentiment_labels",
        "sarcasm_labels",
        "subset_id",
        "source_id",
    )

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        model_features: list[dict[str, Any]] = []
        metadata: dict[str, list[int]] = {key: [] for key in self.metadata_keys}

        for feature in features:
            model_feature = dict(feature)
            for key in self.metadata_keys:
                metadata[key].append(int(model_feature.pop(key)))
            model_features.append(model_feature)

        batch = self.tokenizer.pad(model_features, padding=True, return_tensors="pt")
        batch.update({key: torch.tensor(values, dtype=torch.long) for key, values in metadata.items()})
        return batch


class SharedEncoderTwoHeadClassifier(nn.Module):
    """One text encoder shared by independent sentiment and sarcasm heads."""

    def __init__(
        self,
        backbone: nn.Module,
        pooling: Literal["cls", "last_token"],
        dropout: float = 0.1,
        head_type: Literal["linear", "mlp"] = "linear",
        mlp_hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        hidden_size = int(backbone.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        if head_type == "mlp":
            intermediate_size = mlp_hidden_size or hidden_size // 2
            if not 2 < intermediate_size < hidden_size:
                raise ValueError(
                    "MLP hidden size must be greater than 2 and smaller than the "
                    "encoder hidden size."
                )

            def make_head() -> nn.Sequential:
                return nn.Sequential(
                    nn.Linear(hidden_size, intermediate_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(intermediate_size, 2),
                )

            self.sentiment_head = make_head()
            self.sarcasm_head = make_head()
        elif head_type == "linear":
            self.sentiment_head = nn.Linear(hidden_size, 2)
            self.sarcasm_head = nn.Linear(hidden_size, 2)
        else:
            raise ValueError(f"Unsupported classification head type: {head_type}")

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden_states[:, 0]

        positions = torch.arange(
            attention_mask.shape[1], device=attention_mask.device
        ).unsqueeze(0)
        last_positions = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return hidden_states[batch_indices, last_positions]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )
        pooled = self.dropout(self._pool(outputs.last_hidden_state, attention_mask))
        # Some local checkpoints (notably Qwen) emit bf16 hidden states even
        # outside autocast, while newly created classification heads default to
        # fp32. Linear layers require matching dtypes during evaluation.
        head_dtype = next(self.sentiment_head.parameters()).dtype
        pooled = pooled.to(dtype=head_dtype)
        return {
            "sentiment_logits": self.sentiment_head(pooled),
            "sarcasm_logits": self.sarcasm_head(pooled),
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    """Choose the best available device in CUDA -> MPS -> CPU order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_local_besstie(
    tokenizer: PreTrainedTokenizerBase,
    data_dir: str | Path,
    split_seed: int,
    max_length: int,
    seed: int,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    test_limit: int | None = None,
    include_test: bool = True,
) -> DatasetDict:
    """Load one offline development split and optionally the untouched test CSV."""

    data_path = Path(data_dir)
    split_path = data_path / "splits" / f"seed_{split_seed}"
    csv_paths = {
        "train": split_path / "train.csv",
        "validation": split_path / "validation.csv",
    }
    if include_test:
        csv_paths["test"] = data_path / "test.csv"
    missing = [str(path) for path in csv_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing offline BESSTIE CSV files: "
            + ", ".join(missing)
            + ". Run notebooks/eda.ipynb before copying the project to the HPC."
        )

    required_columns = {
        "example_id",
        "text",
        "source",
        "subset",
        "sentiment",
        "sarcasm",
    }
    frames: dict[str, pd.DataFrame] = {}
    for split, path in csv_paths.items():
        frame = pd.read_csv(path)
        absent = required_columns - set(frame.columns)
        if absent:
            raise ValueError(f"{path} is missing required columns: {sorted(absent)}")
        frames[split] = frame

    id_sets = {split: set(frame["example_id"]) for split, frame in frames.items()}
    if id_sets["train"] & id_sets["validation"]:
        raise ValueError("Training and validation CSVs contain overlapping example IDs.")
    if include_test and (id_sets["train"] | id_sets["validation"]) & id_sets["test"]:
        raise ValueError("Development data overlaps the untouched test CSV.")

    merged = DatasetDict(
        {
            split: Dataset.from_pandas(frame, preserve_index=False)
            for split, frame in frames.items()
        }
    )

    limits = {
        "train": train_limit,
        "validation": validation_limit,
    }
    if include_test:
        limits["test"] = test_limit
    for split, limit in limits.items():
        if limit is not None:
            count = min(limit, len(merged[split]))
            merged[split] = merged[split].shuffle(seed=seed).select(range(count))

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        encoded["sentiment_labels"] = batch["sentiment"]
        encoded["sarcasm_labels"] = batch["sarcasm"]
        encoded["subset_id"] = [SUBSET_TO_ID[value] for value in batch["subset"]]
        encoded["source_id"] = [SOURCE_TO_ID[value.lower()] for value in batch["source"]]
        return encoded

    tokenized = DatasetDict()
    for split, dataset in merged.items():
        tokenized[split] = dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=dataset.column_names,
            desc=f"Tokenising {split}",
        )
    return tokenized


def balanced_class_weights(dataset: Dataset, label_column: str) -> torch.Tensor:
    """Return ordinary inverse-frequency weights for a binary task."""
    counts = np.bincount(
        np.asarray(dataset[label_column], dtype=np.int64), minlength=2
    )
    if np.any(counts == 0):
        raise ValueError(
            f"Both classes must occur in {label_column}; observed counts {counts.tolist()}"
        )
    return torch.tensor(len(dataset) / (2.0 * counts), dtype=torch.float32)


def balanced_subtask_class_weights(
    dataset: Dataset,
    label_column: str,
    variety_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return weights that balance both varieties and both labels for one task."""
    labels = np.asarray(dataset[label_column], dtype=np.int64)
    subset_ids = np.asarray(dataset["subset_id"], dtype=np.int64)
    counts = np.zeros((len(SUBSETS), 2), dtype=np.int64)
    for subset_id in ID_TO_SUBSET:
        counts[subset_id] = np.bincount(labels[subset_ids == subset_id], minlength=2)
    if np.any(counts == 0):
        raise ValueError(
            f"Every variety must contain both classes in {label_column}; "
            f"observed counts {counts.tolist()}"
        )
    if variety_weights is None:
        variety_weights = torch.full((len(SUBSETS),), 1.0 / len(SUBSETS))
    variety_weights_array = np.asarray(variety_weights, dtype=np.float64)
    if variety_weights_array.shape != (len(SUBSETS),):
        raise ValueError(f"Expected {len(SUBSETS)} variety weights.")
    if np.any(variety_weights_array < 0) or not np.isclose(variety_weights_array.sum(), 1.0):
        raise ValueError("Variety weights must be non-negative and sum to one.")
    weights = (
        variety_weights_array[:, None] * len(labels) / (2.0 * counts)
    )
    return torch.tensor(weights, dtype=torch.float32)


def update_adaptive_variety_weights(
    metrics: dict[str, float],
    task: str,
    current_weights: torch.Tensor,
    beta: float,
    temperature: float,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    """Shift weight smoothly toward the lower-F1 variety for one task."""
    if not 0.0 <= beta <= 1.0:
        raise ValueError("adaptive_weight_beta must be between zero and one.")
    if temperature <= 0.0:
        raise ValueError("adaptive_weight_temperature must be positive.")
    if not 0.0 <= minimum <= 0.5 or not 0.5 <= maximum <= 1.0:
        raise ValueError("Variety bounds must bracket 0.5.")
    if not np.isclose(minimum + maximum, 1.0):
        raise ValueError("For two varieties, min and max weights must sum to one.")

    scores = torch.tensor(
        [metrics[f"{subset}_{task}_macro_f1"] for subset in SUBSETS],
        dtype=torch.float32,
    )
    target = torch.softmax(-scores / temperature, dim=0)
    target_first = target[0].clamp(min=minimum, max=maximum)
    target = torch.stack((target_first, 1.0 - target_first))
    updated = (1.0 - beta) * current_weights.cpu() + beta * target
    updated_first = updated[0].clamp(min=minimum, max=maximum)
    return torch.stack((updated_first, 1.0 - updated_first))


def weighted_subtask_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    subset_ids: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Balance a task equally across varieties and labels."""
    example_losses = F.cross_entropy(logits, labels, reduction="none")
    example_weights = weights[subset_ids, labels]
    return (example_losses * example_weights).mean()


def within_variety_class_weights(
    dataset: Dataset, label_column: str
) -> torch.Tensor:
    """Return binary class weights normalised independently within each variety."""
    labels = np.asarray(dataset[label_column], dtype=np.int64)
    subset_ids = np.asarray(dataset["subset_id"], dtype=np.int64)
    counts = np.zeros((len(SUBSETS), 2), dtype=np.int64)
    for subset_id in ID_TO_SUBSET:
        counts[subset_id] = np.bincount(labels[subset_ids == subset_id], minlength=2)
    if np.any(counts == 0):
        raise ValueError(
            f"Every variety must contain both classes in {label_column}; "
            f"observed counts {counts.tolist()}"
        )
    weights = counts.sum(axis=1, keepdims=True) / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32)


def group_dro_task_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    subset_ids: torch.Tensor,
    class_weights: torch.Tensor,
    group_logits: torch.Tensor,
    step_size: float,
) -> torch.Tensor:
    """Compute class-balanced variety losses and update Group DRO weights."""
    example_losses = F.cross_entropy(logits, labels, reduction="none")
    group_losses = torch.zeros(len(SUBSETS), device=logits.device)
    present = torch.zeros(len(SUBSETS), dtype=torch.bool, device=logits.device)

    for subset_id in ID_TO_SUBSET:
        mask = subset_ids == subset_id
        if mask.any():
            example_weights = class_weights[subset_id, labels[mask]]
            group_losses[subset_id] = (
                example_losses[mask] * example_weights
            ).sum() / example_weights.sum()
            present[subset_id] = True

    with torch.no_grad():
        group_logits[present] += step_size * group_losses.detach()[present]
        group_logits -= group_logits.mean()

    group_weights = torch.softmax(group_logits, dim=0)
    active_weights = group_weights * present
    active_weights = active_weights / active_weights.sum()
    return (active_weights * group_losses).sum()


def count_parameters(model: nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_percentage": 100.0 * trainable / total,
    }


def _metrics_for_mask(
    prefix: str,
    mask: np.ndarray,
    sentiment_labels: np.ndarray,
    sentiment_predictions: np.ndarray,
    sarcasm_labels: np.ndarray,
    sarcasm_predictions: np.ndarray,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for task, labels, predictions in (
        ("sentiment", sentiment_labels, sentiment_predictions),
        ("sarcasm", sarcasm_labels, sarcasm_predictions),
    ):
        selected_labels = labels[mask]
        selected_predictions = predictions[mask]
        metrics[f"{prefix}_{task}_macro_f1"] = float(
            f1_score(selected_labels, selected_predictions, labels=[0, 1], average="macro", zero_division=0)
        )
        metrics[f"{prefix}_{task}_accuracy"] = float(
            accuracy_score(selected_labels, selected_predictions)
        )
    return metrics


def search_sarcasm_thresholds_by_variety(
    probabilities: np.ndarray,
    labels: np.ndarray,
    subset_ids: np.ndarray,
    minimum: float = 0.01,
    maximum: float = 0.99,
    steps: int = 99,
) -> dict[str, float]:
    """Choose each variety's validation threshold by sarcasm macro-F1."""
    if not 0.0 < minimum < maximum < 1.0:
        raise ValueError("Sarcasm threshold bounds must lie strictly between 0 and 1.")
    if steps < 2:
        raise ValueError("sarcasm_threshold_steps must be at least two.")
    candidates = np.linspace(minimum, maximum, steps)
    thresholds: dict[str, float] = {}
    for subset_id, subset in ID_TO_SUBSET.items():
        mask = subset_ids == subset_id
        if not mask.any():
            raise ValueError(f"No validation examples found for {subset}.")
        scores = np.asarray(
            [
                f1_score(
                    labels[mask],
                    probabilities[mask] >= threshold,
                    labels=[0, 1],
                    average="macro",
                    zero_division=0,
                )
                for threshold in candidates
            ]
        )
        best_candidates = candidates[np.isclose(scores, scores.max())]
        thresholds[subset] = float(
            best_candidates[np.argmin(np.abs(best_candidates - 0.5))]
        )
    return thresholds


def _sarcasm_predictions_from_thresholds(
    probabilities: np.ndarray,
    subset_ids: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    if set(thresholds) != set(SUBSETS):
        raise ValueError(f"Expected thresholds for {SUBSETS}; received {thresholds}.")
    predictions = np.zeros(len(probabilities), dtype=np.int64)
    for subset_id, subset in ID_TO_SUBSET.items():
        threshold = float(thresholds[subset])
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"Invalid sarcasm threshold for {subset}: {threshold}")
        mask = subset_ids == subset_id
        predictions[mask] = probabilities[mask] >= threshold
    return predictions


def _all_metrics(
    arrays: dict[str, np.ndarray],
    sentiment_predictions: np.ndarray,
    sarcasm_predictions: np.ndarray,
) -> dict[str, float]:
    all_examples = np.ones(len(sentiment_predictions), dtype=bool)
    metrics = _metrics_for_mask(
        "overall",
        all_examples,
        arrays["sentiment_labels"],
        sentiment_predictions,
        arrays["sarcasm_labels"],
        sarcasm_predictions,
    )
    for subset_id, subset in ID_TO_SUBSET.items():
        metrics.update(
            _metrics_for_mask(
                subset,
                arrays["subset_id"] == subset_id,
                arrays["sentiment_labels"],
                sentiment_predictions,
                arrays["sarcasm_labels"],
                sarcasm_predictions,
            )
        )
    for source_id, source in ID_TO_SOURCE.items():
        metrics.update(
            _metrics_for_mask(
                source,
                arrays["source_id"] == source_id,
                arrays["sentiment_labels"],
                sentiment_predictions,
                arrays["sarcasm_labels"],
                sarcasm_predictions,
            )
        )
    metrics["overall_mean_macro_f1"] = (
        metrics["overall_sentiment_macro_f1"]
        + metrics["overall_sarcasm_macro_f1"]
    ) / 2.0
    metrics["selection_score"] = (
        min(
            metrics["en_AU_sentiment_macro_f1"],
            metrics["en_UK_sentiment_macro_f1"],
        )
        + min(
            metrics["en_AU_sarcasm_macro_f1"],
            metrics["en_UK_sarcasm_macro_f1"],
        )
    ) / 2.0
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    sarcasm_thresholds: dict[str, float] | None = None,
    search_sarcasm_thresholds: bool = False,
    threshold_minimum: float = 0.01,
    threshold_maximum: float = 0.99,
    threshold_steps: int = 99,
) -> dict[str, float]:
    if search_sarcasm_thresholds and sarcasm_thresholds is not None:
        raise ValueError("Cannot search for and apply fixed thresholds together.")
    model.eval()
    gathered: dict[str, list[torch.Tensor]] = {
        "sentiment_logits": [],
        "sarcasm_logits": [],
        "sentiment_labels": [],
        "sarcasm_labels": [],
        "subset_id": [],
        "source_id": [],
    }

    for batch in dataloader:
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
        }
        sentiment_labels = batch.pop("sentiment_labels")
        sarcasm_labels = batch.pop("sarcasm_labels")
        subset_id = batch.pop("subset_id")
        source_id = batch.pop("source_id")
        outputs = model(**batch)

        values = {
            "sentiment_logits": outputs["sentiment_logits"],
            "sarcasm_logits": outputs["sarcasm_logits"],
            "sentiment_labels": sentiment_labels,
            "sarcasm_labels": sarcasm_labels,
            "subset_id": subset_id,
            "source_id": source_id,
        }
        for key, value in values.items():
            gathered[key].append(value.cpu())

    arrays = {key: torch.cat(values).numpy() for key, values in gathered.items()}
    sentiment_predictions = arrays["sentiment_logits"].argmax(axis=1)
    default_sarcasm_predictions = arrays["sarcasm_logits"].argmax(axis=1)
    default_metrics = _all_metrics(
        arrays, sentiment_predictions, default_sarcasm_predictions
    )
    if not search_sarcasm_thresholds and sarcasm_thresholds is None:
        return default_metrics

    shifted_logits = arrays["sarcasm_logits"] - arrays["sarcasm_logits"].max(
        axis=1, keepdims=True
    )
    exponentiated = np.exp(shifted_logits)
    sarcasm_probabilities = exponentiated[:, 1] / exponentiated.sum(axis=1)
    if search_sarcasm_thresholds:
        sarcasm_thresholds = search_sarcasm_thresholds_by_variety(
            probabilities=sarcasm_probabilities,
            labels=arrays["sarcasm_labels"],
            subset_ids=arrays["subset_id"],
            minimum=threshold_minimum,
            maximum=threshold_maximum,
            steps=threshold_steps,
        )
    if sarcasm_thresholds is None:
        raise RuntimeError("Threshold evaluation requested without thresholds.")
    sarcasm_predictions = _sarcasm_predictions_from_thresholds(
        sarcasm_probabilities, arrays["subset_id"], sarcasm_thresholds
    )
    metrics = _all_metrics(arrays, sentiment_predictions, sarcasm_predictions)
    for subset in SUBSETS:
        metrics[f"sarcasm_threshold_{subset}"] = float(sarcasm_thresholds[subset])
        metrics[f"default_threshold_{subset}_sarcasm_macro_f1"] = default_metrics[
            f"{subset}_sarcasm_macro_f1"
        ]
    metrics["default_threshold_overall_sarcasm_macro_f1"] = default_metrics[
        "overall_sarcasm_macro_f1"
    ]
    metrics["default_threshold_overall_mean_macro_f1"] = default_metrics[
        "overall_mean_macro_f1"
    ]
    metrics["default_threshold_selection_score"] = default_metrics[
        "selection_score"
    ]
    metrics["threshold_selection_score_gain"] = (
        metrics["selection_score"] - default_metrics["selection_score"]
    )
    return metrics


def save_metadata(
    config: TrainingConfig,
    metrics: dict[str, Any],
    parameter_counts: dict[str, int | float],
    method: str,
    selected_epoch: int,
) -> None:
    metadata = {
        "method": method,
        "selected_epoch": selected_epoch,
        "training_config": asdict(config),
        "parameter_counts": parameter_counts,
        "validation_metrics": metrics,
        "label_mapping": {
            "sentiment": {"0": "negative", "1": "positive"},
            "sarcasm": {"0": "not_sarcastic", "1": "sarcastic"},
        },
    }
    summary_dir = Path(config.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def save_checkpoint(
    model: nn.Module,
    output_dir: Path,
    config: TrainingConfig,
    metrics: dict[str, float],
    parameter_counts: dict[str, int | float],
    method: str,
    selected_epoch: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(
        output_dir / "backbone",
        safe_serialization=True,
    )
    torch.save(
        {
            "sentiment_head": model.sentiment_head.state_dict(),
            "sarcasm_head": model.sarcasm_head.state_dict(),
        },
        output_dir / "classification_heads.pt",
    )
    save_metadata(
        config=config,
        metrics=metrics,
        parameter_counts=parameter_counts,
        method=method,
        selected_epoch=selected_epoch,
    )


def train_multitask(
    model: SharedEncoderTwoHeadClassifier,
    tokenizer: PreTrainedTokenizerBase,
    datasets: DatasetDict,
    config: TrainingConfig,
    method: str,
) -> dict[str, Any]:
    seed_everything(config.seed)
    if config.early_stopping_patience is not None and config.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be at least one or None.")
    if config.group_dro_step_size < 0.0:
        raise ValueError("group_dro_step_size must be non-negative.")
    if config.search_sarcasm_thresholds:
        if not 0.0 < config.sarcasm_threshold_min < config.sarcasm_threshold_max < 1.0:
            raise ValueError(
                "Sarcasm threshold search bounds must lie strictly between 0 and 1."
            )
        if config.sarcasm_threshold_steps < 2:
            raise ValueError("sarcasm_threshold_steps must be at least two.")
    device = select_device()
    use_mixed_precision = device.type == "cuda" and config.mixed_precision != "no"
    autocast_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
    use_grad_scaler = use_mixed_precision and config.mixed_precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    if config.mixed_precision != "no" and device.type != "cuda":
        print(f"Mixed precision is CUDA-only in this pipeline; using full precision on {device.type}.")

    if config.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable()
        if hasattr(model.backbone, "enable_input_require_grads"):
            model.backbone.enable_input_require_grads()

    collator = MultitaskDataCollator(tokenizer)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        datasets["validation"],
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = None
    if config.evaluate_test:
        test_loader = DataLoader(
            datasets["test"],
            batch_size=config.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )

    parameter_counts = count_parameters(model)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    update_steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_update_steps = update_steps_per_epoch * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_update_steps * config.warmup_ratio),
        num_training_steps=total_update_steps,
    )

    model.to(device)
    adaptive_weighting = config.loss_weighting == "adaptive_task_variety_class"
    group_dro_weighting = config.loss_weighting == "group_dro"
    variety_weights = {
        "sentiment": torch.full((len(SUBSETS),), 1.0 / len(SUBSETS)),
        "sarcasm": torch.full((len(SUBSETS),), 1.0 / len(SUBSETS)),
    }
    group_dro_logits = {
        "sentiment": torch.zeros(len(SUBSETS), device=device),
        "sarcasm": torch.zeros(len(SUBSETS), device=device),
    }
    if config.loss_weighting == "task_class":
        sentiment_weights = balanced_class_weights(
            datasets["train"], "sentiment_labels"
        ).to(device)
        sarcasm_weights = balanced_class_weights(
            datasets["train"], "sarcasm_labels"
        ).to(device)
    elif group_dro_weighting:
        sentiment_weights = within_variety_class_weights(
            datasets["train"], "sentiment_labels"
        ).to(device)
        sarcasm_weights = within_variety_class_weights(
            datasets["train"], "sarcasm_labels"
        ).to(device)
    else:
        sentiment_weights = balanced_subtask_class_weights(
            datasets["train"], "sentiment_labels", variety_weights["sentiment"]
        ).to(device)
        sarcasm_weights = balanced_subtask_class_weights(
            datasets["train"], "sarcasm_labels", variety_weights["sarcasm"]
        ).to(device)

    print(json.dumps({"device": str(device), "parameter_counts": parameter_counts}, indent=2))
    print(
        f"Class weights ({config.loss_weighting}):",
        {"sentiment": sentiment_weights.tolist(), "sarcasm": sarcasm_weights.tolist()},
    )

    output_dir = Path(config.output_dir)
    best_score = -math.inf
    best_epoch: int | None = None
    best_validation_metrics: dict[str, float] | None = None
    best_trainable_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    optimizer.zero_grad()

    for epoch in range(1, config.epochs + 1):
        model.train()
        cumulative_loss = 0.0
        completed_batches = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{config.epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch_index, batch in enumerate(progress_bar):
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            subset_ids = batch.pop("subset_id")
            batch.pop("source_id")
            sentiment_labels = batch.pop("sentiment_labels")
            sarcasm_labels = batch.pop("sarcasm_labels")

            group_start = (
                batch_index // config.gradient_accumulation_steps
            ) * config.gradient_accumulation_steps
            accumulation_size = min(
                config.gradient_accumulation_steps,
                len(train_loader) - group_start,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=use_mixed_precision,
            ):
                outputs = model(**batch)
                if group_dro_weighting:
                    sentiment_loss = group_dro_task_loss(
                        outputs["sentiment_logits"],
                        sentiment_labels,
                        subset_ids,
                        sentiment_weights,
                        group_dro_logits["sentiment"],
                        config.group_dro_step_size,
                    )
                    sarcasm_loss = group_dro_task_loss(
                        outputs["sarcasm_logits"],
                        sarcasm_labels,
                        subset_ids,
                        sarcasm_weights,
                        group_dro_logits["sarcasm"],
                        config.group_dro_step_size,
                    )
                elif config.loss_weighting != "task_class":
                    sentiment_loss = weighted_subtask_cross_entropy(
                        outputs["sentiment_logits"],
                        sentiment_labels,
                        subset_ids,
                        sentiment_weights,
                    )
                    sarcasm_loss = weighted_subtask_cross_entropy(
                        outputs["sarcasm_logits"],
                        sarcasm_labels,
                        subset_ids,
                        sarcasm_weights,
                    )
                else:
                    sentiment_loss = F.cross_entropy(
                        outputs["sentiment_logits"], sentiment_labels, weight=sentiment_weights
                    )
                    sarcasm_loss = F.cross_entropy(
                        outputs["sarcasm_logits"], sarcasm_labels, weight=sarcasm_weights
                    )
                loss = 0.5 * (sentiment_loss + sarcasm_loss)
                scaled_loss = loss / accumulation_size

            scaler.scale(scaled_loss).backward()
            should_update = (
                (batch_index + 1) % config.gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            cumulative_loss += float(loss.detach())
            completed_batches += 1
            progress_bar.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                learning_rate=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        metrics = evaluate(
            model,
            validation_loader,
            device,
            search_sarcasm_thresholds=config.search_sarcasm_thresholds,
            threshold_minimum=config.sarcasm_threshold_min,
            threshold_maximum=config.sarcasm_threshold_max,
            threshold_steps=config.sarcasm_threshold_steps,
        )
        weights_used = {
            task: {
                subset: float(weights[subset_id])
                for subset_id, subset in ID_TO_SUBSET.items()
            }
            for task, weights in variety_weights.items()
        }
        next_variety_weights = variety_weights
        if adaptive_weighting and epoch < config.epochs:
            adaptive_metrics = dict(metrics)
            if config.search_sarcasm_thresholds:
                for subset in SUBSETS:
                    adaptive_metrics[f"{subset}_sarcasm_macro_f1"] = metrics[
                        f"default_threshold_{subset}_sarcasm_macro_f1"
                    ]
            next_variety_weights = {
                task: update_adaptive_variety_weights(
                    metrics=adaptive_metrics,
                    task=task,
                    current_weights=variety_weights[task],
                    beta=config.adaptive_weight_beta,
                    temperature=config.adaptive_weight_temperature,
                    minimum=config.min_variety_weight,
                    maximum=config.max_variety_weight,
                )
                for task in ("sentiment", "sarcasm")
            }

        epoch_result = {
            "epoch": epoch,
            "train_loss": cumulative_loss / max(completed_batches, 1),
            "training_variety_weights": weights_used,
            **metrics,
        }
        if group_dro_weighting:
            epoch_result["group_dro_weights"] = {
                task: {
                    subset: float(torch.softmax(logits, dim=0)[subset_id])
                    for subset_id, subset in ID_TO_SUBSET.items()
                }
                for task, logits in group_dro_logits.items()
            }
        if adaptive_weighting and epoch < config.epochs:
            epoch_result["next_epoch_variety_weights"] = {
                task: {
                    subset: float(weights[subset_id])
                    for subset_id, subset in ID_TO_SUBSET.items()
                }
                for task, weights in next_variety_weights.items()
            }
        history.append(epoch_result)

        print(json.dumps(epoch_result, indent=2, sort_keys=True))

        selection_key = (
            "selection_score"
            if config.selection_metric == "min_variety_mean_macro_f1"
            else "overall_mean_macro_f1"
        )
        if metrics[selection_key] > best_score:
            best_score = metrics[selection_key]
            best_epoch = epoch
            best_validation_metrics = dict(metrics)
            epochs_without_improvement = 0
            if config.evaluate_test:
                best_trainable_state = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                }
            if config.save_model_checkpoint:
                save_checkpoint(
                    model,
                    output_dir / "best",
                    config,
                    metrics,
                    parameter_counts,
                    method,
                    selected_epoch=epoch,
                )
            else:
                save_metadata(
                    config=config,
                    metrics=metrics,
                    parameter_counts=parameter_counts,
                    method=method,
                    selected_epoch=epoch,
                )
        else:
            epochs_without_improvement += 1

        if adaptive_weighting and epoch < config.epochs:
            variety_weights = next_variety_weights
            sentiment_weights = balanced_subtask_class_weights(
                datasets["train"],
                "sentiment_labels",
                variety_weights["sentiment"],
            ).to(device)
            sarcasm_weights = balanced_subtask_class_weights(
                datasets["train"],
                "sarcasm_labels",
                variety_weights["sarcasm"],
            ).to(device)

        if (
            config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            print(
                f"Early stopping after epoch {epoch}; no validation improvement "
                f"for {epochs_without_improvement} epoch(s)."
            )
            break

    if best_epoch is None or best_validation_metrics is None:
        raise RuntimeError("Training completed without selecting a best checkpoint.")

    test_metrics: dict[str, float] | None = None
    if config.evaluate_test:
        if best_trainable_state is None or test_loader is None:
            raise RuntimeError("Test evaluation requested without a restorable model state.")
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(
                        best_trainable_state[name].to(
                            device=parameter.device, dtype=parameter.dtype
                        )
                    )
        selected_sarcasm_thresholds = None
        if config.search_sarcasm_thresholds:
            selected_sarcasm_thresholds = {
                subset: float(
                    best_validation_metrics[f"sarcasm_threshold_{subset}"]
                )
                for subset in SUBSETS
            }
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            sarcasm_thresholds=selected_sarcasm_thresholds,
        )

    summary_dir = Path(config.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (summary_dir / "validation_result.json").write_text(
        json.dumps(
            {
                "selected_epoch": best_epoch,
                "best_validation_score": best_score,
                "selected_sarcasm_thresholds": (
                    {
                        subset: best_validation_metrics[
                            f"sarcasm_threshold_{subset}"
                        ]
                        for subset in SUBSETS
                    }
                    if config.search_sarcasm_thresholds
                    else None
                ),
                "validation_metrics": best_validation_metrics,
            },
            indent=2,
        )
        + "\n"
    )
    if config.evaluate_test:
        (summary_dir / "test_metrics.json").write_text(
            json.dumps(
                {"selected_epoch": best_epoch, "test_metrics": test_metrics},
                indent=2,
            )
            + "\n"
        )
        metadata_path = summary_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["test_metrics"] = test_metrics
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(
            json.dumps(
                {"selected_epoch": best_epoch, "test_metrics": test_metrics},
                indent=2,
            )
        )
    else:
        print(
            json.dumps(
                {
                    "selected_epoch": best_epoch,
                    "best_validation_score": best_score,
                    "test_evaluated": False,
                },
                indent=2,
            )
        )

    return {
        "best_validation_score": best_score,
        "best_validation_mean_macro_f1": best_score,
        "best_epoch": best_epoch,
        "validation_metrics": best_validation_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "parameter_counts": parameter_counts,
        "selected_sarcasm_thresholds": (
            {
                subset: best_validation_metrics[f"sarcasm_threshold_{subset}"]
                for subset in SUBSETS
            }
            if config.search_sarcasm_thresholds
            else None
        ),
    }


def run_training(
    config: TrainingConfig,
    tokenizer: PreTrainedTokenizerBase,
    model_factory: Callable[[], SharedEncoderTwoHeadClassifier],
    method: str,
) -> dict[str, Any]:
    seed_everything(config.seed)
    datasets = load_local_besstie(
        tokenizer=tokenizer,
        data_dir=config.data_dir,
        split_seed=config.split_seed,
        max_length=config.max_length,
        seed=config.seed,
        train_limit=config.train_limit,
        validation_limit=config.validation_limit,
        test_limit=config.test_limit,
        include_test=config.evaluate_test,
    )
    model = model_factory()
    return train_multitask(model, tokenizer, datasets, config, method)
