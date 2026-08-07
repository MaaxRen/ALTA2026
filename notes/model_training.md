# Multitask Baseline Training

The project provides two training entry points:

- `scripts/train_roberta.py`: full fine-tuning of `FacebookAI/roberta-base`.
- `scripts/train_qwen_lora.py`: LoRA fine-tuning of `Qwen/Qwen3-Embedding-0.6B`.

Both models use one shared text embedder and two independent binary classification heads:

- Sentiment: negative (`0`) or positive (`1`).
- Sarcasm: not sarcastic (`0`) or sarcastic (`1`).

The joint objective is the mean of the two class-weighted cross-entropy losses. Class weights are calculated from the training split. Model selection uses the mean of validation sentiment and sarcasm macro-F1.

Two additional entry points target the official minimum-variety score without changing
the baseline scripts:

- `scripts/train_roberta_minvariety.py`
- `scripts/train_qwen_lora_minvariety.py`

They select the best epoch using

```text
(min(sentiment_en_AU, sentiment_en_UK)
 + min(sarcasm_en_AU, sarcasm_en_UK)) / 2
```

where each component is macro-F1 on the newly created validation split. Their
inverse-frequency example weights balance every task × variety × label cell. This
keeps sentiment and sarcasm equally weighted while preventing the easier variety
from dominating either task. The weighting is based only on the training split.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
```

The HPC must not download the dataset. Run `notebooks/eda.ipynb` on an internet-connected machine first, then copy the complete project directory, including `data/besstie`, to the HPC.

The local data layout is:

```text
data/besstie/
├── all.csv
├── original_train.csv
├── test.csv
├── split_manifest.csv
└── splits/
    ├── seed_2026/{train.csv,validation.csv}
    ├── seed_2027/{train.csv,validation.csv}
    ├── seed_2028/{train.csv,validation.csv}
    ├── seed_2029/{train.csv,validation.csv}
    └── seed_2030/{train.csv,validation.csv}
```

`test.csv` is the original Hugging Face validation set and remains untouched during training and model selection. Each seeded development split divides the original training set into 85% training and 15% validation data.

The scripts load their staged weights from `resources/pretrained_model/roberta` and `resources/pretrained_model/qwen3-embedding-0.6b`. Both set `LOCAL_FILES_ONLY = True`, preventing accidental network access. This large model directory is excluded from Git, so it must be copied to the HPC separately from a Git clone.

All training settings are constants near the top of each script. There are no command-line training arguments. Device selection is automatic, with the explicit priority:

```text
CUDA -> MPS -> CPU
```

CUDA uses the configured `bf16`, `fp16`, or full precision mode. MPS and CPU use full precision for portability during local testing.

## RoBERTa full fine-tuning

Edit the configuration block in `scripts/train_roberta.py`, then run:

```bash
python scripts/train_roberta.py
```

If memory is constrained, reduce `TRAIN_BATCH_SIZE` and increase `GRADIENT_ACCUMULATION_STEPS` in the script to retain the desired effective batch size.

## Qwen embedding LoRA fine-tuning

Edit the configuration block in `scripts/train_qwen_lora.py`, then run:

```bash
python scripts/train_qwen_lora.py
```

Qwen uses the model card's left-padding and last-token pooling convention. LoRA is applied to all linear layers in the embedding backbone. The two classification heads remain fully trainable.

Each command runs seeds 2026–2030 sequentially. To run fewer splits while testing, edit `SPLIT_SEEDS`, for example:

```python
SPLIT_SEEDS = (2026, 2028)
```

## Smoke runs

For a quick local or HPC smoke test, temporarily use these values in the relevant script:

```python
SPLIT_SEEDS = (2026,)
EPOCHS = 1
TRAIN_LIMIT = 64
VALIDATION_LIMIT = 32
TEST_LIMIT = 32
```

Then run the script normally with `python scripts/<script_name>.py`.

## Outputs

Model checkpoints contain only loadable model artifacts:

```text
model_checkpoints/<model>/seed_*/best/
├── backbone/                 # Full RoBERTa weights or Qwen LoRA adapter
└── classification_heads.pt  # Sentiment and sarcasm heads
```

All run summaries are kept separately:

```text
training_summary/runs/<model>/
├── all_results.json
├── all_results.csv
├── seed_2026/
│   ├── metadata.json
│   ├── history.json
│   └── test_metrics.json
├── seed_2027/...
└── seed_2028/...
```

The best epoch is selected exclusively using the mean of sentiment and sarcasm macro-F1 on the newly created validation set. Its trainable parameters are restored before one test evaluation. Metrics include macro-F1 and accuracy overall, by English variety, and by source (`google` or `reddit`). Source-specific metrics are important because the label distributions differ substantially between the two domains.

The preceding selection rule applies to the two baseline scripts. The two
`*_minvariety.py` scripts instead use the minimum-variety formula above and save to
separate `*_minvariety` checkpoint and summary directories.

## Adaptive minimum-variety training

Two further entry points update the en_AU/en_UK loss allocation after each
validation epoch:

- `scripts/train_roberta_adaptive.py`
- `scripts/train_qwen_lora_adaptive.py`

Run them in the same way as the other scripts. They still keep sentiment and
sarcasm at 50% each, but use a temperature-scaled softmax of negative validation
macro-F1 to give more of each task's next-epoch weight to its weaker variety. An
EMA (`beta=0.3`) smooths updates, and each variety is bounded to 25%–75%. Label
classes remain inverse-frequency balanced inside each task–variety group.

Every `history.json` epoch records `training_variety_weights` and, when another
epoch remains, `next_epoch_variety_weights`. Adaptive checkpoints and summaries
use separate `*_adaptive_minvariety` directories.

### Adaptive MLP heads

The following variants replace each direct linear classifier with an independent
one-hidden-layer MLP while retaining the shared encoder and adaptive objective:

- `scripts/train_roberta_adaptive_mlp.py`
- `scripts/train_qwen_lora_adaptive_mlp.py`

Each head uses `hidden_size -> hidden_size // 2 -> 2`, with GELU and dropout after
the hidden layer. Their outputs use separate directories ending in
`_adaptive_minvariety_mlp`.

## Six-run comparison

All values are three-seed means on the untouched test set.

| ID | Backbone model | Training strategy | F1 sentiment en AU | F1 sentiment en UK | F1 sarcasm en AU | F1 sarcasm en UK | Score |
|---|---|---|---:|---:|---:|---:|---:|
| R1 | RoBERTa | Full FT, static, linear heads | 0.9132 | 0.9497 | 0.7670 | 0.6445 | 0.7789 |
| R2 | RoBERTa | Full FT, adaptive, linear heads | 0.9152 | 0.9445 | 0.7362 | 0.6508 | 0.7830 |
| R3 | RoBERTa | Full FT, adaptive, MLP heads | 0.9090 | 0.9410 | 0.7647 | 0.6809 | 0.7949 |
| Q1 | Qwen3-Embedding-0.6B | LoRA, static, linear heads | 0.9206 | 0.9497 | 0.7774 | 0.6699 | 0.7953 |
| Q2 | Qwen3-Embedding-0.6B | LoRA, adaptive, linear heads | 0.9195 | 0.9567 | 0.7543 | 0.6874 | **0.8034** |
| Q3 | Qwen3-Embedding-0.6B | LoRA, adaptive, MLP heads | 0.9133 | 0.9575 | 0.7339 | 0.6930 | 0.8031 |

Key findings:

- Qwen adaptive linear and MLP are effectively tied.
- RoBERTa benefits most from adaptive MLP heads: +0.0160 over static.
- en_AU sentiment and en_UK sarcasm are always the score minima.
- en_UK sarcasm remains the main bottleneck.
- Qwen results vary more across splits than RoBERTa results.

Comparability:

- Equal: 5 epochs, effective batch 16, splits, seeds, max length 256, dropout
  0.1, weight decay 0.01, warmup 0.1, and BF16 configuration.
- Model-specific: RoBERTa full fine-tuning at `2e-5`; Qwen LoRA at `2e-4`.
- Static and adaptive runs intentionally differ in loss and checkpoint metric.
- A clean weighting ablation still needs static variety balancing with
  minimum-score selection.

Plots and reproducible analysis code are in
`notebooks/training_results_analysis.ipynb`.

## Hyperparameter tuning

Run:

```bash
python scripts/tune_hyperparameters.py
```

Default search:

- Qwen LoRA, 48 deterministic random configurations.
- Five development splits: 2026–2030.
- Up to 8 epochs; early-stopping patience 2.
- Minimum-variety validation score for every configuration.
- No test loading, test evaluation, or model checkpoint saving.
- Resumes completed configuration–split runs automatically.

Outputs:

```text
training_summary/hyperparameter_tuning/multitask_random_v1/qwen/
├── search_manifest.json
├── leaderboard.csv
├── leaderboard.json
└── cfg_*/
    ├── configuration.json
    ├── aggregate.json
    └── seed_*/{history.json,metadata.json,validation_result.json}
```

To tune another standard Hugging Face model:

1. Add one entry to `MODEL_REGISTRY`.
2. Add its name to `MODELS_TO_TUNE`.
3. Change `SEARCH_NAME` for a new independent search.

Select hyperparameters from validation means only. Run the untouched test set
after freezing the winning configuration.

## Separate task models

Entry points:

- `scripts/train_roberta_separate_tasks.py`
- `scripts/train_qwen_lora_separate_tasks.py`

For each split:

- Train independent sentiment and sarcasm backbones/adapters.
- Balance variety × label cells within each task.
- Select each task checkpoint by its minimum en_AU/en_UK validation macro-F1.
- Combine the two selected task scores only for the final benchmark score.

Artifacts use `*_separate_tasks` directories. Each task checkpoint contains one
backbone/adapter and `classification_head.pt`.

## Group DRO

Entry points:

- `scripts/train_roberta_group_dro.py`
- `scripts/train_qwen_lora_group_dro.py`

The shared model retains two heads. For each task, Group DRO:

- Computes class-balanced en_AU and en_UK training losses.
- Updates variety weights by exponentiated gradient with step size `0.01`.
- Averages the robust sentiment and sarcasm losses 50/50.
- Uses no validation feedback for loss weighting.
- Selects checkpoints using the minimum-variety validation score.

Epoch histories record `group_dro_weights`. Artifacts use `*_group_dro`
directories.
