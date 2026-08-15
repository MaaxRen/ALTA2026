"""File-based prompt inference helpers for Gemma BESSTIE runs."""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import redirect_stdout
import importlib.util
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

import pandas as pd

from alta2026.prompt_templates import (
    FewShotExample,
    SYSTEM_INSTRUCTION,
    build_few_shot_prompt,
    build_single_shot_prompt,
)


OFFICIAL_SCORER_PATH = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "alta_official"
    / "alta_2026_scoring_program"
    / "evaluate.py"
)
_OFFICIAL_SPEC = importlib.util.spec_from_file_location(
    "alta_2026_official_prompt_evaluate", OFFICIAL_SCORER_PATH
)
if _OFFICIAL_SPEC is None or _OFFICIAL_SPEC.loader is None:
    raise ImportError(f"Could not load official scorer: {OFFICIAL_SCORER_PATH}")
_OFFICIAL_MODULE = importlib.util.module_from_spec(_OFFICIAL_SPEC)
_OFFICIAL_SPEC.loader.exec_module(_OFFICIAL_MODULE)
official_evaluate = _OFFICIAL_MODULE.evaluate

RunPromptBuilder = Callable[[pd.Series], str]

SUBMISSION_COLUMNS = ["source", "variety", "text", "sentiment", "sarcasm"]
_JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class ParsedConfidence:
    sentiment_confidence: float
    sarcasm_confidence: float

    @property
    def sentiment_prediction(self) -> int:
        return int(self.sentiment_confidence >= 0.5)

    @property
    def sarcasm_prediction(self) -> int:
        return int(self.sarcasm_confidence >= 0.5)


def _clamp_probability(value: Any, field_name: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric, received {value!r}") from exc
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"{field_name} must lie in [0, 1], received {probability!r}"
        )
    return probability


def parse_confidence_payload(text: str) -> ParsedConfidence:
    stripped = text.strip()
    candidate = stripped
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if "\n" in stripped:
            stripped = stripped.split("\n", 1)[1]
        candidate = stripped.rsplit("\n", 1)[0].strip()
    else:
        match = _JSON_BLOCK_PATTERN.search(text)
        if match is not None:
            candidate = match.group(0)
    payload = json.loads(candidate)
    return ParsedConfidence(
        sentiment_confidence=_clamp_probability(
            payload.get("sentiment_confidence"), "sentiment_confidence"
        ),
        sarcasm_confidence=_clamp_probability(
            payload.get("sarcasm_confidence"), "sarcasm_confidence"
        ),
    )


class GemmaApiClient:
    """Minimal REST client for Google-hosted Gemma generation."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
    ) -> None:
        try:
            import httpx
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Gemma API calls require `httpx` in the active Python environment."
            ) from exc
        self.httpx = httpx
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.client = httpx.Client(timeout=timeout_seconds)
        self.timeout_seconds = timeout_seconds
        self.url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )

    @classmethod
    def from_environment(
        cls,
        model: str,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
        env_var: str = "GEMMA_API_KEY",
    ) -> "GemmaApiClient":
        api_key = os.environ.get(env_var)
        if not api_key:
            raise EnvironmentError(
                f"Missing environment variable {env_var}. Export it locally before running."
            )
        return cls(
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )

    def generate_json(self, prompt: str, max_retries: int = 3) -> str:
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
            },
        }
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.post(
                    self.url,
                    headers={"x-goog-api-key": self.api_key},
                    json=payload,
                )
            except (
                self.httpx.ReadTimeout,
                self.httpx.ConnectTimeout,
                self.httpx.TimeoutException,
                self.httpx.TransportError,
            ) as exc:
                if attempt < max_retries:
                    time.sleep(2 ** (attempt - 1))
                    continue
                raise RuntimeError(
                    "Gemma API request failed after retries due to timeout or "
                    f"transport error: {exc}. Client timeout was "
                    f"{self.timeout_seconds} seconds."
                ) from exc
            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                time.sleep(2 ** (attempt - 1))
                continue
            if response.is_error:
                raise RuntimeError(
                    "Gemma API request failed with "
                    f"HTTP {response.status_code}: {response.text}"
                )
            body = response.json()
            candidates = body.get("candidates", [])
            if not candidates:
                raise ValueError(f"Model response had no candidates: {body}")
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(str(part.get("text", "")) for part in parts)
            if not text.strip():
                raise ValueError(f"Model response had no text payload: {body}")
            return text
        raise RuntimeError("Gemma API retries exhausted unexpectedly.")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_completed_predictions(path: Path) -> dict[str, dict[str, Any]]:
    return {str(record["example_id"]): record for record in load_jsonl_records(path)}


def build_answer_frame(
    evaluation_frame: pd.DataFrame,
    prediction_records: dict[str, dict[str, Any]],
    sentiment_threshold: float = 0.5,
    sarcasm_threshold: float = 0.5,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in evaluation_frame.iterrows():
        example_id = str(row["example_id"])
        if example_id not in prediction_records:
            raise KeyError(f"Missing prediction for example_id={example_id}")
        prediction = prediction_records[example_id]
        rows.append(
            {
                "source": row["source"],
                "variety": row["variety"],
                "text": row["text"],
                "sentiment": int(
                    float(prediction["sentiment_confidence"]) >= sentiment_threshold
                ),
                "sarcasm": int(
                    float(prediction["sarcasm_confidence"]) >= sarcasm_threshold
                ),
            }
        )
    return pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)


def score_answer_frame(
    answer_frame: pd.DataFrame,
    gold_frame: pd.DataFrame,
) -> dict[str, Any]:
    gold = gold_frame[["variety", "sentiment", "sarcasm"]].copy()
    run = answer_frame[["variety", "sentiment", "sarcasm"]].copy()
    with TemporaryDirectory(prefix="alta2026_prompt_score_") as temporary_dir:
        temporary_path = Path(temporary_dir)
        gold_path = temporary_path / "truth.csv"
        run_path = temporary_path / "answer.csv"
        gold.to_csv(gold_path, index=False)
        run.to_csv(run_path, index=False)
        with redirect_stdout(StringIO()):
            scores, final_score = official_evaluate(run_path, gold_path)
    return {
        "component_scores": {key: float(value) for key, value in scores.items()},
        "final_score": float(final_score),
    }


def default_few_shot_examples(
    train_frame: pd.DataFrame,
    shots: int = 4,
) -> list[FewShotExample]:
    selections = [
        ("en_AU", "google", 1, 0),
        ("en_AU", "reddit", 0, 1),
        ("en_AU", "reddit", 1, 0),
        ("en_UK", "google", 0, 0),
        ("en_UK", "reddit", 0, 1),
        ("en_UK", "reddit", 1, 0),
    ]
    if shots < 1:
        raise ValueError("shots must be at least 1.")
    selected_targets = selections[:shots]
    examples: list[FewShotExample] = []
    for subset, source, sentiment, sarcasm in selected_targets:
        matches = train_frame[
            (train_frame["subset"] == subset)
            & (train_frame["source"] == source)
            & (train_frame["sentiment"] == sentiment)
            & (train_frame["sarcasm"] == sarcasm)
        ]
        if matches.empty:
            continue
        row = matches.iloc[0]
        examples.append(
            FewShotExample(
                text=str(row["text"]),
                variety=str(row["variety"]),
                source=str(row["source"]),
                sentiment=int(row["sentiment"]),
                sarcasm=int(row["sarcasm"]),
            )
        )
    if not examples:
        raise ValueError("Could not construct any few-shot examples from training data.")
    if len(examples) != len(selected_targets):
        raise ValueError(
            "Could not construct the requested few-shot set from training data. "
            f"Requested {len(selected_targets)} targets, found {len(examples)}."
        )
    return examples


def run_prompt_inference(
    evaluation_frame: pd.DataFrame,
    run_dir: Path,
    client: GemmaApiClient,
    prompt_builder: RunPromptBuilder,
    sentiment_threshold: float = 0.5,
    sarcasm_threshold: float = 0.5,
    sleep_seconds: float = 0.0,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    requests_path = run_dir / "prompt_requests.jsonl"
    responses_path = run_dir / "responses.jsonl"
    errors_path = run_dir / "errors.jsonl"
    answer_path = run_dir / "answer.csv"
    parsed_predictions = load_completed_predictions(responses_path)

    for _, row in evaluation_frame.iterrows():
        example_id = str(row["example_id"])
        if example_id in parsed_predictions:
            continue
        prompt = prompt_builder(row)
        append_jsonl(
            requests_path,
            {
                "example_id": example_id,
                "source": row["source"],
                "variety": row["variety"],
                "prompt": prompt,
            },
        )
        raw_response = client.generate_json(prompt)
        try:
            parsed = parse_confidence_payload(raw_response)
        except Exception as exc:  # noqa: BLE001
            append_jsonl(
                errors_path,
                {
                    "example_id": example_id,
                    "error": str(exc),
                    "raw_response": raw_response,
                },
            )
            raise
        record = {
            "example_id": example_id,
            "sentiment_confidence": parsed.sentiment_confidence,
            "sarcasm_confidence": parsed.sarcasm_confidence,
            "raw_response": raw_response,
        }
        append_jsonl(responses_path, record)
        parsed_predictions[example_id] = record
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    answer_frame = build_answer_frame(
        evaluation_frame,
        parsed_predictions,
        sentiment_threshold=sentiment_threshold,
        sarcasm_threshold=sarcasm_threshold,
    )
    answer_frame.to_csv(answer_path, index=False)
    score_payload = score_answer_frame(answer_frame, evaluation_frame)
    score_payload.update(
        {
            "answer_csv": str(answer_path),
            "prompt_requests_jsonl": str(requests_path),
            "responses_jsonl": str(responses_path),
            "errors_jsonl": str(errors_path),
            "rows_scored": int(len(answer_frame)),
            "sentiment_threshold": sentiment_threshold,
            "sarcasm_threshold": sarcasm_threshold,
        }
    )
    (run_dir / "scores.json").write_text(json.dumps(score_payload, indent=2) + "\n")
    return score_payload


def make_single_shot_builder() -> RunPromptBuilder:
    def build(row: pd.Series) -> str:
        return build_single_shot_prompt(
            text=str(row["text"]),
            variety=str(row["variety"]),
            source=str(row["source"]),
        )

    return build


def make_few_shot_builder(examples: list[FewShotExample]) -> RunPromptBuilder:
    def build(row: pd.Series) -> str:
        return build_few_shot_prompt(
            text=str(row["text"]),
            variety=str(row["variety"]),
            source=str(row["source"]),
            examples=examples,
        )

    return build
