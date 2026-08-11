#!/usr/bin/env python3
"""Evaluate one multitask checkpoint with the official ALTA 2026 scorer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from alta2026.multitask_training import (  # noqa: E402
    SharedEncoderTwoHeadClassifier,
    select_device,
)
from resources.alta_official.alta_2026_scoring_program.evaluate import (  # noqa: E402
    evaluate as official_evaluate,
)

# Edit this configuration block. No command-line arguments are required.
CHECKPOINT_DIR = (
    REPO_ROOT
    / "model_checkpoints"
    / "roberta_adaptive_minvariety"
    / "seed_2026"
    / "best"
)
VALIDATION_CSV = (
    REPO_ROOT
    / "resources"
    / "alta_official"
    / "alta2026_public_data"
    / "valid.csv"
)
ANSWER_TEMPLATE_CSV = REPO_ROOT / "resources" / "alta_official" / "answer.csv"
OUTPUT_DIR = REPO_ROOT / "training_summary" / "official_evaluation"

# Usually inferred. Set either path explicitly for an unfamiliar checkpoint.
BASE_MODEL_DIR: Path | None = None
TOKENIZER_DIR: Path | None = None
POOLING: str | None = None

LOCAL_FILES_ONLY = True
MAX_LENGTH = 256
EVAL_BATCH_SIZE = 16

# Optional frozen validation-selected thresholds, using internal variety names.
# Example: {"en_AU": 0.46, "en_UK": 0.39}
SARCASM_THRESHOLDS: dict[str, float] | None = None

SUBMISSION_COLUMNS = ["source", "variety", "text", "sentiment", "sarcasm"]
DIALECT_TO_SUBSET = {"en-AU": "en_AU", "en-UK": "en_UK"}


def _local_model_directories() -> list[Path]:
    root = REPO_ROOT / "resources" / "pretrained_model"
    return sorted(path for path in root.iterdir() if path.is_dir())


def _resolve_base_model(adapter_dir: Path) -> Path:
    if BASE_MODEL_DIR is not None:
        return BASE_MODEL_DIR
    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    recorded = Path(adapter_config["base_model_name_or_path"])
    candidates = [recorded, REPO_ROOT / "resources" / "pretrained_model" / recorded.name]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not resolve the LoRA base model locally. Set BASE_MODEL_DIR in "
        f"this script. Recorded base: {recorded}"
    )


def _resolve_tokenizer(backbone_dir: Path, model_type: str, base_dir: Path | None) -> Path:
    if TOKENIZER_DIR is not None:
        return TOKENIZER_DIR
    if base_dir is not None:
        return base_dir
    for candidate in _local_model_directories():
        config_path = candidate / "config.json"
        if not config_path.is_file() or not (candidate / "tokenizer.json").is_file():
            continue
        if json.loads(config_path.read_text()).get("model_type") == model_type:
            return candidate
    raise FileNotFoundError(
        f"No staged tokenizer matches model type {model_type!r}. Set TOKENIZER_DIR."
    )


def _head_settings(head_states: dict[str, dict[str, torch.Tensor]]) -> tuple[str, int | None]:
    sentiment_state = head_states["sentiment_head"]
    if "weight" in sentiment_state:
        return "linear", None
    if "0.weight" in sentiment_state:
        return "mlp", int(sentiment_state["0.weight"].shape[0])
    raise ValueError(
        f"Unrecognised classification head state keys: {sorted(sentiment_state)}"
    )


def load_checkpoint() -> tuple[SharedEncoderTwoHeadClassifier, object, torch.device]:
    backbone_dir = CHECKPOINT_DIR / "backbone"
    heads_path = CHECKPOINT_DIR / "classification_heads.pt"
    if not backbone_dir.is_dir() or not heads_path.is_file():
        raise FileNotFoundError(
            f"Expected backbone/ and classification_heads.pt under {CHECKPOINT_DIR}"
        )

    device = select_device()
    adapter_checkpoint = (backbone_dir / "adapter_config.json").is_file()
    base_dir = _resolve_base_model(backbone_dir) if adapter_checkpoint else None
    config_source = base_dir if base_dir is not None else backbone_dir
    backbone_config = AutoConfig.from_pretrained(
        str(config_source), local_files_only=LOCAL_FILES_ONLY
    )
    model_type = str(backbone_config.model_type)
    tokenizer_dir = _resolve_tokenizer(backbone_dir, model_type, base_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        use_fast=True,
        padding_side="right" if model_type == "roberta" else "left",
        local_files_only=LOCAL_FILES_ONLY,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_checkpoint:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise SystemExit("Loading a LoRA checkpoint requires PEFT.") from exc
        model_kwargs = {"local_files_only": LOCAL_FILES_ONLY}
        if device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16
        if model_type == "qwen3":
            model_kwargs["attn_implementation"] = "sdpa"
        backbone = AutoModel.from_pretrained(str(base_dir), **model_kwargs)
        backbone = PeftModel.from_pretrained(
            backbone, str(backbone_dir), is_trainable=False
        )
    else:
        backbone = AutoModel.from_pretrained(
            str(backbone_dir), local_files_only=LOCAL_FILES_ONLY
        )

    head_states = torch.load(heads_path, map_location="cpu", weights_only=True)
    head_type, mlp_hidden_size = _head_settings(head_states)
    pooling = POOLING or ("cls" if model_type == "roberta" else "last_token")
    model = SharedEncoderTwoHeadClassifier(
        backbone=backbone,
        pooling=pooling,
        dropout=0.0,
        head_type=head_type,
        mlp_hidden_size=mlp_hidden_size,
    )
    model.sentiment_head.load_state_dict(head_states["sentiment_head"])
    model.sarcasm_head.load_state_dict(head_states["sarcasm_head"])
    model.to(device).eval()
    return model, tokenizer, device


@torch.no_grad()
def predict(
    model: SharedEncoderTwoHeadClassifier,
    tokenizer: object,
    device: torch.device,
    validation: pd.DataFrame,
) -> tuple[list[int], list[int]]:
    sentiment_predictions: list[int] = []
    sarcasm_predictions: list[int] = []
    for start in tqdm(
        range(0, len(validation), EVAL_BATCH_SIZE),
        desc="Official validation inference",
        unit="batch",
    ):
        batch_frame = validation.iloc[start : start + EVAL_BATCH_SIZE]
        batch = tokenizer(
            batch_frame["text"].tolist(),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        sentiment_predictions.extend(
            outputs["sentiment_logits"].argmax(dim=1).cpu().tolist()
        )

        if SARCASM_THRESHOLDS is None:
            sarcasm_batch = outputs["sarcasm_logits"].argmax(dim=1).cpu()
        else:
            probabilities = outputs["sarcasm_logits"].softmax(dim=1)[:, 1].cpu()
            thresholds = torch.tensor(
                [
                    SARCASM_THRESHOLDS[DIALECT_TO_SUBSET[variety]]
                    for variety in batch_frame["variety"]
                ]
            )
            sarcasm_batch = (probabilities >= thresholds).long()
        sarcasm_predictions.extend(sarcasm_batch.tolist())
    return sentiment_predictions, sarcasm_predictions


def main() -> None:
    validation = pd.read_csv(VALIDATION_CSV)
    absent = set(SUBMISSION_COLUMNS) - set(validation.columns)
    if absent:
        raise ValueError(f"Validation CSV is missing columns: {sorted(absent)}")
    if not validation["variety"].isin(DIALECT_TO_SUBSET).all():
        raise ValueError("Validation varieties must be en-AU or en-UK.")

    template = pd.read_csv(ANSWER_TEMPLATE_CSV)
    if list(template.columns) != SUBMISSION_COLUMNS:
        raise ValueError(
            f"Unexpected answer template columns: {list(template.columns)}"
        )
    descriptor_columns = ["source", "variety", "text"]
    if not validation[descriptor_columns].equals(template[descriptor_columns]):
        raise ValueError("answer.csv rows do not align with the official validation CSV.")

    model, tokenizer, device = load_checkpoint()
    sentiment, sarcasm = predict(model, tokenizer, device, validation)
    answer = validation[descriptor_columns].copy()
    answer["sentiment"] = sentiment
    answer["sarcasm"] = sarcasm
    answer = answer[SUBMISSION_COLUMNS]

    run_name = CHECKPOINT_DIR.parents[1].name
    seed_name = CHECKPOINT_DIR.parent.name
    output_dir = OUTPUT_DIR / run_name / seed_name
    output_dir.mkdir(parents=True, exist_ok=True)
    answer_path = output_dir / "answer.csv"
    answer.to_csv(answer_path, index=False)

    component_scores, final_score = official_evaluate(answer_path, VALIDATION_CSV)
    result = {
        "checkpoint": str(CHECKPOINT_DIR),
        "validation_csv": str(VALIDATION_CSV),
        "answer_csv": str(answer_path),
        "device": str(device),
        "component_scores": component_scores,
        "final_score": final_score,
        "sarcasm_thresholds": SARCASM_THRESHOLDS,
    }
    result_path = output_dir / "official_scores.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved submission: {answer_path}")
    print(f"Saved official scores: {result_path}")


if __name__ == "__main__":
    main()
