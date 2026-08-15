#!/usr/bin/env python3
"""Run few-shot Gemma inference on BESSTIE evaluation data."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from alta2026.prompt_inference import (  # noqa: E402
    GemmaApiClient,
    default_few_shot_examples,
    make_few_shot_builder,
    run_prompt_inference,
)


DEFAULT_TRAIN_CSV = REPO_ROOT / "data" / "besstie" / "original_train.csv"
DEFAULT_EVAL_CSV = REPO_ROOT / "data" / "besstie" / "test.csv"
DEFAULT_RUN_ROOT = REPO_ROOT / "training_summary" / "prompt_runs" / "gemma_few_shot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--evaluation-csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--model", required=True, help="Gemma model id exposed by the API.")
    parser.add_argument("--api-key-env", default="GEMMA_API_KEY")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--shots",
        type=int,
        default=4,
        help="Number of deterministic few-shot examples to prepend.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--sentiment-threshold", type=float, default=0.5)
    parser.add_argument("--sarcasm-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train_csv)
    evaluation = pd.read_csv(args.evaluation_csv)
    if args.limit is not None:
        evaluation = evaluation.iloc[: args.limit].copy()

    run_dir = args.run_dir
    if run_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = DEFAULT_RUN_ROOT / timestamp

    examples = default_few_shot_examples(train, shots=args.shots)
    client = GemmaApiClient.from_environment(
        model=args.model,
        temperature=args.temperature,
        env_var=args.api_key_env,
    )
    payload = run_prompt_inference(
        evaluation_frame=evaluation,
        run_dir=run_dir,
        client=client,
        prompt_builder=make_few_shot_builder(examples),
        sentiment_threshold=args.sentiment_threshold,
        sarcasm_threshold=args.sarcasm_threshold,
        sleep_seconds=args.sleep_seconds,
    )
    manifest = {
        "mode": "few_shot",
        "model": args.model,
        "train_csv": str(args.train_csv),
        "evaluation_csv": str(args.evaluation_csv),
        "rows": int(len(evaluation)),
        "shots": args.shots,
        "few_shot_examples": [example.__dict__ for example in examples],
        "score": payload["final_score"],
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
