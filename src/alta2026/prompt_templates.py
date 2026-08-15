"""Prompt builders for Gemma-based BESSTIE inference."""

from __future__ import annotations

import json
from dataclasses import dataclass


PROMPT_OUTPUT_SCHEMA = {
    "sentiment_confidence": "float in [0, 1], confidence that sentiment is positive (label 1)",
    "sarcasm_confidence": "float in [0, 1], confidence that sarcasm is sarcastic (label 1)",
}


SYSTEM_INSTRUCTION = (
    "You are labeling one text example for a shared task. "
    "Return only valid JSON with no markdown fences or extra commentary."
)


@dataclass(frozen=True)
class FewShotExample:
    text: str
    variety: str
    source: str
    sentiment: int
    sarcasm: int


def _task_instructions() -> str:
    return (
        "Predict two binary properties for the input text.\n"
        "Return only valid JSON.\n"
        "Both `sentiment_confidence` and `sarcasm_confidence` must be JSON floats in [0, 1],\n"
        "for example `0.0`, `0.37`, or `1.0`.\n"
        "Interpretation:\n"
        "- `sentiment_confidence` is confidence that sentiment = 1 (positive). Label 0 = negative.\n"
        "- `sarcasm_confidence` is confidence that sarcasm = 1 (sarcastic). Label 0 = not sarcastic.\n"
        "- `0.0` means very confident the label is 0.\n"
        "- `0.5` means uncertain or borderline.\n"
        "- `1.0` means very confident the label is 1.\n"
        "The JSON schema is:\n"
        f"{json.dumps(PROMPT_OUTPUT_SCHEMA, indent=2)}"
    )


def build_single_shot_prompt(text: str, variety: str, source: str) -> str:
    return (
        f"{_task_instructions()}\n\n"
        f"Metadata:\n- variety: {variety}\n- source: {source}\n\n"
        f"Text:\n{text}\n"
    )


def build_few_shot_prompt(
    text: str,
    variety: str,
    source: str,
    examples: list[FewShotExample],
) -> str:
    demonstrations: list[str] = []
    for index, example in enumerate(examples, start=1):
        demonstrations.append(
            "\n".join(
                [
                    f"Example {index}",
                    f"variety: {example.variety}",
                    f"source: {example.source}",
                    f"text: {example.text}",
                    "output:",
                    json.dumps(
                        {
                            "sentiment_confidence": float(example.sentiment),
                            "sarcasm_confidence": float(example.sarcasm),
                        }
                    ),
                ]
            )
        )
    return (
        f"{_task_instructions()}\n\n"
        "Here are labeled examples. Match the same output style.\n\n"
        + "\n\n".join(demonstrations)
        + "\n\nNow label the next example.\n"
        + f"Metadata:\n- variety: {variety}\n- source: {source}\n\n"
        + f"Text:\n{text}\n"
    )
