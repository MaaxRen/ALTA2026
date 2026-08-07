#!/usr/bin/env python3
"""Train the shared RoBERTa multitask model with per-task variety Group DRO."""

from group_dro_training_common import run_group_dro_training


if __name__ == "__main__":
    run_group_dro_training("roberta")
