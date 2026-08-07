#!/usr/bin/env python3
"""Run adaptive minimum-variety LoRA fine-tuning for Qwen."""

from adaptive_training_common import run_adaptive_training


if __name__ == "__main__":
    run_adaptive_training("qwen")
