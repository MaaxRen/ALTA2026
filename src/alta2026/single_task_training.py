"""Single-task training utilities for separate BESSTIE task models."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup

from alta2026.multitask_training import (
    ID_TO_SOURCE,
    ID_TO_SUBSET,
    MultitaskDataCollator,
    TrainingConfig,
    count_parameters,
    load_local_besstie,
    seed_everything,
    select_device,
    balanced_subtask_class_weights,
    weighted_subtask_cross_entropy,
)

TaskName = Literal["sentiment", "sarcasm"]


class SingleTaskClassifier(nn.Module):
    """One transformer backbone and one binary task head."""

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
        if head_type == "linear":
            self.classification_head = nn.Linear(hidden_size, 2)
        elif head_type == "mlp":
            intermediate_size = mlp_hidden_size or hidden_size // 2
            if not 2 < intermediate_size < hidden_size:
                raise ValueError(
                    "MLP hidden size must be greater than 2 and smaller than the "
                    "encoder hidden size."
                )
            self.classification_head = nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_size, 2),
            )
        else:
            raise ValueError(f"Unsupported classification head type: {head_type}")

    def _pool(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden_states[:, 0]
        positions = torch.arange(
            attention_mask.shape[1], device=attention_mask.device
        ).unsqueeze(0)
        last_positions = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return hidden_states[batch_indices, last_positions]

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )
        pooled = self.dropout(self._pool(outputs.last_hidden_state, attention_mask))
        head_dtype = next(self.classification_head.parameters()).dtype
        return self.classification_head(pooled.to(dtype=head_dtype))


@torch.no_grad()
def evaluate_single_task(
    model: SingleTaskClassifier,
    dataloader: DataLoader,
    device: torch.device,
    task: TaskName,
) -> dict[str, float]:
    model.eval()
    logits, labels, subset_ids, source_ids = [], [], [], []
    target_key = f"{task}_labels"

    for batch in dataloader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        target = batch.pop(target_key)
        batch.pop("sarcasm_labels" if task == "sentiment" else "sentiment_labels")
        subset_id = batch.pop("subset_id")
        source_id = batch.pop("source_id")
        logits.append(model(**batch).cpu())
        labels.append(target.cpu())
        subset_ids.append(subset_id.cpu())
        source_ids.append(source_id.cpu())

    predictions = torch.cat(logits).numpy().argmax(axis=1)
    labels_array = torch.cat(labels).numpy()
    subset_array = torch.cat(subset_ids).numpy()
    source_array = torch.cat(source_ids).numpy()

    def metrics_for_mask(prefix: str, mask: np.ndarray) -> dict[str, float]:
        return {
            f"{prefix}_macro_f1": float(
                f1_score(
                    labels_array[mask],
                    predictions[mask],
                    labels=[0, 1],
                    average="macro",
                    zero_division=0,
                )
            ),
            f"{prefix}_accuracy": float(
                accuracy_score(labels_array[mask], predictions[mask])
            ),
        }

    metrics = metrics_for_mask("overall", np.ones(len(labels_array), dtype=bool))
    for subset_id, subset in ID_TO_SUBSET.items():
        metrics.update(metrics_for_mask(subset, subset_array == subset_id))
    for source_id, source in ID_TO_SOURCE.items():
        metrics.update(metrics_for_mask(source, source_array == source_id))
    metrics["selection_score"] = min(
        metrics["en_AU_macro_f1"], metrics["en_UK_macro_f1"]
    )
    return metrics


def save_single_task_checkpoint(
    model: SingleTaskClassifier,
    output_dir: Path,
    config: TrainingConfig,
    task: TaskName,
    method: str,
    selected_epoch: int,
    validation_metrics: dict[str, float],
    parameter_counts: dict[str, int | float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(output_dir / "backbone", safe_serialization=True)
    torch.save(
        model.classification_head.state_dict(),
        output_dir / "classification_head.pt",
    )
    summary_dir = Path(config.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "task": task,
        "method": method,
        "selected_epoch": selected_epoch,
        "training_config": asdict(config),
        "parameter_counts": parameter_counts,
        "validation_metrics": validation_metrics,
        "label_mapping": (
            {"0": "negative", "1": "positive"}
            if task == "sentiment"
            else {"0": "not_sarcastic", "1": "sarcastic"}
        ),
    }
    (summary_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def train_single_task(
    model: SingleTaskClassifier,
    tokenizer: PreTrainedTokenizerBase,
    datasets: DatasetDict,
    config: TrainingConfig,
    task: TaskName,
    method: str,
) -> dict[str, Any]:
    seed_everything(config.seed)
    device = select_device()
    use_mixed_precision = device.type == "cuda" and config.mixed_precision != "no"
    autocast_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
    use_grad_scaler = use_mixed_precision and config.mixed_precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    if config.mixed_precision != "no" and device.type != "cuda":
        print(f"Mixed precision is CUDA-only; using full precision on {device.type}.")
    if config.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable()
        if hasattr(model.backbone, "enable_input_require_grads"):
            model.backbone.enable_input_require_grads()

    collator = MultitaskDataCollator(tokenizer)
    loaders = {
        split: DataLoader(
            datasets[split],
            batch_size=(
                config.train_batch_size if split == "train" else config.eval_batch_size
            ),
            shuffle=split == "train",
            collate_fn=collator,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )
        for split in ("train", "validation", "test")
    }

    parameter_counts = count_parameters(model)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    update_steps_per_epoch = math.ceil(
        len(loaders["train"]) / config.gradient_accumulation_steps
    )
    total_update_steps = update_steps_per_epoch * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_update_steps * config.warmup_ratio),
        num_training_steps=total_update_steps,
    )

    model.to(device)
    label_column = f"{task}_labels"
    class_weights = balanced_subtask_class_weights(
        datasets["train"], label_column
    ).to(device)
    print(
        json.dumps(
            {"task": task, "device": str(device), "parameter_counts": parameter_counts},
            indent=2,
        )
    )

    best_score = -math.inf
    best_epoch = None
    best_state = None
    best_validation_metrics = None
    history = []
    optimizer.zero_grad()

    for epoch in range(1, config.epochs + 1):
        model.train()
        cumulative_loss = 0.0
        completed_batches = 0
        progress = tqdm(
            loaders["train"],
            desc=f"{task} epoch {epoch}/{config.epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch_index, batch in enumerate(progress):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            labels = batch.pop(label_column)
            batch.pop("sarcasm_labels" if task == "sentiment" else "sentiment_labels")
            subset_ids = batch.pop("subset_id")
            batch.pop("source_id")

            group_start = (
                batch_index // config.gradient_accumulation_steps
            ) * config.gradient_accumulation_steps
            accumulation_size = min(
                config.gradient_accumulation_steps,
                len(loaders["train"]) - group_start,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=use_mixed_precision,
            ):
                logits = model(**batch)
                loss = weighted_subtask_cross_entropy(
                    logits, labels, subset_ids, class_weights
                )
                scaled_loss = loss / accumulation_size

            scaler.scale(scaled_loss).backward()
            should_update = (
                (batch_index + 1) % config.gradient_accumulation_steps == 0
                or batch_index + 1 == len(loaders["train"])
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
            progress.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                learning_rate=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        metrics = evaluate_single_task(model, loaders["validation"], device, task)
        epoch_result = {
            "epoch": epoch,
            "train_loss": cumulative_loss / max(completed_batches, 1),
            **metrics,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, indent=2, sort_keys=True))

        if metrics["selection_score"] > best_score:
            best_score = metrics["selection_score"]
            best_epoch = epoch
            best_validation_metrics = dict(metrics)
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            save_single_task_checkpoint(
                model=model,
                output_dir=Path(config.output_dir) / "best",
                config=config,
                task=task,
                method=method,
                selected_epoch=epoch,
                validation_metrics=metrics,
                parameter_counts=parameter_counts,
            )

    if best_epoch is None or best_state is None or best_validation_metrics is None:
        raise RuntimeError(f"No best checkpoint selected for {task}.")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(
                    best_state[name].to(device=parameter.device, dtype=parameter.dtype)
                )
    test_metrics = evaluate_single_task(model, loaders["test"], device, task)

    summary_dir = Path(config.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (summary_dir / "test_metrics.json").write_text(
        json.dumps(
            {"selected_epoch": best_epoch, "test_metrics": test_metrics}, indent=2
        )
        + "\n"
    )
    metadata_path = summary_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["test_metrics"] = test_metrics
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    return {
        "task": task,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "validation_metrics": best_validation_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "parameter_counts": parameter_counts,
    }


def run_single_task_training(
    config: TrainingConfig,
    tokenizer: PreTrainedTokenizerBase,
    model_factory: Callable[[], SingleTaskClassifier],
    task: TaskName,
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
    )
    return train_single_task(
        model_factory(), tokenizer, datasets, config, task=task, method=method
    )
