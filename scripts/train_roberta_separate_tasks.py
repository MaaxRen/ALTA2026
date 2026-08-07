#!/usr/bin/env python3
"""Train separate RoBERTa models for sentiment and sarcasm."""

from separate_task_training_common import run_separate_task_training


if __name__ == "__main__":
    run_separate_task_training("roberta")
