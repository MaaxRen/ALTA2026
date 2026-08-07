# ALTA 2026 — BESSTIE

Internal workshop project using the `unswnlporg/BESSTIE` dataset. The work focuses on the `en_AU` and `en_UK` subsets and jointly predicts:

- sentiment: negative or positive
- sarcasm: not sarcastic or sarcastic

The evaluation score gives equal importance to the two tasks and uses the weaker English variety for each task:

```text
score = (min(sentiment_AU, sentiment_UK)
       + min(sarcasm_AU, sarcasm_UK)) / 2
```

All components are macro-F1 scores.

## Project layout

```text
data/besstie/                 Offline dataset and five seeded splits
model_checkpoints/            Saved model checkpoints
notebooks/                    EDA and training-result analysis
notes/                        Dataset-paper and training notes
resources/papers/             BESSTIE paper
resources/pretrained_model/   Local pretrained models (not tracked by Git)
scripts/                      Training and hyperparameter-tuning scripts
src/alta2026/                 Shared training code
training_summary/             Run results and tuning summaries
```

## Data

`notebooks/eda.ipynb` downloads and filters the dataset, creates the plots, and writes the data needed for offline HPC training.

The original BESSTIE validation set is retained as the test set. The original training set is divided into 85% training and 15% validation using seeds 2026–2030. Model and epoch selection use only these new validation sets.

## Models

The main backbones are:

- RoBERTa with full fine-tuning
- Qwen3-Embedding-0.6B with LoRA fine-tuning

Experiments include shared linear or MLP classification heads, adaptive variety weighting, separate sentiment and sarcasm models, and Group DRO. Each training entry point runs all five data splits and uses the device priority CUDA → MPS → CPU.

Hyperparameter tuning is resumable and stores its results separately under `training_summary/hyperparameter_tuning/`. It uses validation data only and does not save model checkpoints.

## Outputs

Model weights are stored under `model_checkpoints/`. Metrics, histories, test predictions, and aggregate results are stored separately under `training_summary/runs/`.

See `notes/model_training.md` for the experiment summary and `notes/dataset_paper_info.md` for notes on the dataset paper.
