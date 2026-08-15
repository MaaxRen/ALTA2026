"""POS-based train-to-test experiments using existing BESSTIE POS summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from alta2026.pos_features import SUMMARY_TAGS
from alta2026.prompt_inference import score_answer_frame

LABEL_COLUMNS = ("sentiment", "sarcasm")
METADATA_COLUMNS = (
    "example_id",
    "source",
    "subset",
    "variety",
    "text",
    "sentiment",
    "sarcasm",
)
POS_COUNT_COLUMNS = ("token_count",) + SUMMARY_TAGS
RATE_FEATURE_COLUMNS = tuple(f"{tag.lower()}_rate" for tag in SUMMARY_TAGS)
DERIVED_FEATURE_COLUMNS = (
    "adj_minus_verb_rate",
    "noun_minus_pron_rate",
    "adjective_to_verb_ratio",
    "noun_to_pron_ratio",
)
BASE_FEATURE_COLUMNS = RATE_FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS
CENTROID_DISTANCE_FEATURE_COLUMNS = (
    "distance_to_class_0",
    "distance_to_class_1",
    "distance_margin_1_minus_0",
)
PREDICTION_COLUMNS = ("source", "variety", "text", "sentiment", "sarcasm")
EXPERIMENT_ORDER = (
    "experiment_a_logreg",
    "experiment_b_nearest_centroid",
    "experiment_c_distance_logreg",
    "experiment_c_combined_logreg",
)
RATIO_EPSILON = 1e-6
LOGISTIC_RANDOM_STATE = 2026


@dataclass(frozen=True)
class DatasetBundle:
    """Aligned gold data plus POS-derived features for a split."""

    gold: pd.DataFrame
    features: pd.DataFrame


@dataclass(frozen=True)
class StandardizationStats:
    """Train-only feature normalization parameters."""

    means: pd.Series
    stds: pd.Series

    def transform(self, frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
        return (frame.loc[:, columns] - self.means.loc[list(columns)]) / self.stds.loc[
            list(columns)
        ]


@dataclass(frozen=True)
class CentroidResult:
    """Centroids and per-example distances for one binary task."""

    centroids: pd.DataFrame
    distance_features: pd.DataFrame


def load_pos_dataset(gold_csv: Path, pos_summary_csv: Path) -> DatasetBundle:
    """Join an original gold CSV with its POS summary and validate alignment."""
    gold = pd.read_csv(gold_csv)
    pos_summary = pd.read_csv(pos_summary_csv)
    if "text" not in gold.columns:
        raise ValueError(f"Expected `text` column in {gold_csv}")

    merged = gold.merge(
        pos_summary[
            [
                "example_id",
                "source",
                "subset",
                "variety",
                "sentiment",
                "sarcasm",
                *POS_COUNT_COLUMNS,
            ]
        ],
        on=["example_id", "source", "subset", "variety", "sentiment", "sarcasm"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(gold):
        raise ValueError(
            "Gold CSV and POS summary did not align one-to-one. "
            f"gold={len(gold)} pos={len(pos_summary)} merged={len(merged)}"
        )
    merged = merged.loc[:, [*METADATA_COLUMNS, *POS_COUNT_COLUMNS]]
    return DatasetBundle(gold=gold.loc[:, METADATA_COLUMNS].copy(), features=merged)


def build_feature_table(pos_frame: pd.DataFrame) -> pd.DataFrame:
    """Create additive POS feature columns from existing summary counts."""
    feature_table = pos_frame.copy()
    token_counts = feature_table["token_count"].astype(float).replace(0.0, np.nan)
    for tag in SUMMARY_TAGS:
        rate_column = f"{tag.lower()}_rate"
        feature_table[rate_column] = feature_table[tag].astype(float) / token_counts
    feature_table["adj_minus_verb_rate"] = (
        feature_table["adj_rate"] - feature_table["verb_rate"]
    )
    feature_table["noun_minus_pron_rate"] = (
        feature_table["noun_rate"] - feature_table["pron_rate"]
    )
    feature_table["adjective_to_verb_ratio"] = (
        feature_table["adj_rate"] + RATIO_EPSILON
    ) / (feature_table["verb_rate"] + RATIO_EPSILON)
    feature_table["noun_to_pron_ratio"] = (
        feature_table["noun_rate"] + RATIO_EPSILON
    ) / (feature_table["pron_rate"] + RATIO_EPSILON)
    return feature_table.fillna(0.0)


def fit_standardization(
    train_features: pd.DataFrame, columns: tuple[str, ...]
) -> StandardizationStats:
    """Fit train-only z-score normalization stats."""
    means = train_features.loc[:, columns].mean(axis=0)
    stds = train_features.loc[:, columns].std(axis=0, ddof=0).replace(0.0, 1.0)
    return StandardizationStats(means=means, stds=stds)


def _task_metrics(
    labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray | None = None
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)
        ),
    }
    if probabilities is not None:
        metrics["mean_positive_probability"] = float(np.mean(probabilities))
    return metrics


def fit_logistic_regression(
    train_matrix: pd.DataFrame, labels: pd.Series
) -> LogisticRegression:
    """Fit a small, deterministic binary logistic model."""
    model = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        random_state=LOGISTIC_RANDOM_STATE,
        max_iter=1000,
    )
    model.fit(train_matrix, labels.astype(int))
    return model


def logistic_predictions(
    model: LogisticRegression, matrix: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = model.predict_proba(matrix)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int64)
    return predictions, probabilities


def coefficient_table(
    model: LogisticRegression, feature_names: tuple[str, ...]
) -> pd.DataFrame:
    """Return coefficients sorted by absolute magnitude."""
    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.coef_[0],
        }
    )
    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
    return coefficients.sort_values(
        by=["abs_coefficient", "feature"], ascending=[False, True]
    ).reset_index(drop=True)


def compute_task_centroids(
    train_matrix: pd.DataFrame,
    train_labels: pd.Series,
    test_matrix: pd.DataFrame,
) -> CentroidResult:
    """Compute normalized class centroids and test distances for a binary task."""
    train_values = train_matrix.to_numpy(dtype=float)
    test_values = test_matrix.to_numpy(dtype=float)
    labels = train_labels.to_numpy(dtype=int)
    centroids: dict[int, np.ndarray] = {}
    for class_label in (0, 1):
        class_values = train_values[labels == class_label]
        if len(class_values) == 0:
            raise ValueError(f"No train examples for class {class_label}.")
        centroids[class_label] = class_values.mean(axis=0)

    distance_to_0 = np.linalg.norm(test_values - centroids[0], axis=1)
    distance_to_1 = np.linalg.norm(test_values - centroids[1], axis=1)
    centroid_frame = pd.DataFrame(
        [
            {"class_label": class_label, **dict(zip(train_matrix.columns, centroid))}
            for class_label, centroid in centroids.items()
        ]
    )
    distance_features = pd.DataFrame(
        {
            "distance_to_class_0": distance_to_0,
            "distance_to_class_1": distance_to_1,
            "distance_margin_1_minus_0": distance_to_1 - distance_to_0,
        }
    )
    return CentroidResult(
        centroids=centroid_frame,
        distance_features=distance_features,
    )


def nearest_centroid_predictions(distance_features: pd.DataFrame) -> np.ndarray:
    """Predict the class with the smaller centroid distance."""
    return (
        distance_features["distance_to_class_1"].to_numpy()
        < distance_features["distance_to_class_0"].to_numpy()
    ).astype(np.int64)


def evaluate_multitask_predictions(
    gold_frame: pd.DataFrame,
    sentiment_predictions: np.ndarray,
    sarcasm_predictions: np.ndarray,
) -> dict[str, Any]:
    """Score combined predictions with sklearn metrics and the official scorer."""
    answer_frame = gold_frame.loc[:, ["source", "variety", "text"]].copy()
    answer_frame["sentiment"] = sentiment_predictions.astype(int)
    answer_frame["sarcasm"] = sarcasm_predictions.astype(int)

    official_scores = score_answer_frame(answer_frame, gold_frame)
    sentiment_metrics = _task_metrics(
        gold_frame["sentiment"].to_numpy(dtype=int),
        sentiment_predictions.astype(int),
    )
    sarcasm_metrics = _task_metrics(
        gold_frame["sarcasm"].to_numpy(dtype=int),
        sarcasm_predictions.astype(int),
    )
    return {
        "task_metrics": {
            "sentiment": sentiment_metrics,
            "sarcasm": sarcasm_metrics,
        },
        "component_scores": official_scores["component_scores"],
        "final_score": official_scores["final_score"],
        "answer_frame": answer_frame,
    }


def run_all_experiments(
    train_bundle: DatasetBundle,
    test_bundle: DatasetBundle,
    output_dir: Path,
) -> dict[str, Any]:
    """Run Experiments A, B, and C and persist review-friendly artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_features = build_feature_table(train_bundle.features)
    test_features = build_feature_table(test_bundle.features)
    train_features.to_csv(output_dir / "original_train_features.csv", index=False)
    test_features.to_csv(output_dir / "test_features.csv", index=False)

    normalization = fit_standardization(train_features, BASE_FEATURE_COLUMNS)
    normalization_frame = pd.DataFrame(
        {
            "feature": BASE_FEATURE_COLUMNS,
            "train_mean": normalization.means.loc[list(BASE_FEATURE_COLUMNS)].to_numpy(),
            "train_std": normalization.stds.loc[list(BASE_FEATURE_COLUMNS)].to_numpy(),
        }
    )
    normalization_frame.to_csv(output_dir / "normalization_stats.csv", index=False)

    train_base_matrix = normalization.transform(train_features, BASE_FEATURE_COLUMNS)
    test_base_matrix = normalization.transform(test_features, BASE_FEATURE_COLUMNS)

    experiment_predictions: dict[str, dict[str, np.ndarray]] = {}
    experiment_results: dict[str, Any] = {}
    centroid_distance_store: dict[str, pd.DataFrame] = {}

    for task in LABEL_COLUMNS:
        task_output_dir = output_dir / "task_artifacts" / task
        task_output_dir.mkdir(parents=True, exist_ok=True)
        train_labels = train_features[task]
        test_labels = test_features[task]

        experiment_a_dir = task_output_dir / "experiment_a_logreg"
        experiment_a_dir.mkdir(parents=True, exist_ok=True)
        experiment_a_model = fit_logistic_regression(train_base_matrix, train_labels)
        experiment_a_predictions, experiment_a_probabilities = logistic_predictions(
            experiment_a_model, test_base_matrix
        )
        coefficient_table(
            experiment_a_model, BASE_FEATURE_COLUMNS
        ).to_csv(experiment_a_dir / "coefficients.csv", index=False)
        pd.DataFrame(
            {
                "example_id": test_features["example_id"],
                "label": test_labels,
                "prediction": experiment_a_predictions,
                "positive_probability": experiment_a_probabilities,
            }
        ).to_csv(experiment_a_dir / "predictions.csv", index=False)
        experiment_predictions.setdefault("experiment_a_logreg", {})[task] = (
            experiment_a_predictions
        )

        centroid_result = compute_task_centroids(
            train_base_matrix, train_labels, test_base_matrix
        )
        centroid_result.centroids.to_csv(
            task_output_dir / "centroids.csv", index=False
        )
        centroid_result.distance_features.to_csv(
            task_output_dir / "test_centroid_distances.csv", index=False
        )
        centroid_distance_store[task] = centroid_result.distance_features.copy()

        experiment_b_predictions = nearest_centroid_predictions(
            centroid_result.distance_features
        )
        pd.DataFrame(
            {
                "example_id": test_features["example_id"],
                "label": test_labels,
                **centroid_result.distance_features.to_dict(orient="list"),
                "prediction": experiment_b_predictions,
            }
        ).to_csv(task_output_dir / "experiment_b_predictions.csv", index=False)
        experiment_predictions.setdefault("experiment_b_nearest_centroid", {})[task] = (
            experiment_b_predictions
        )

        distance_train = compute_task_centroids(
            train_base_matrix, train_labels, train_base_matrix
        ).distance_features
        distance_test = centroid_result.distance_features

        experiment_c_distance_dir = task_output_dir / "experiment_c_distance_logreg"
        experiment_c_distance_dir.mkdir(parents=True, exist_ok=True)
        experiment_c_distance_model = fit_logistic_regression(distance_train, train_labels)
        experiment_c_distance_predictions, experiment_c_distance_probabilities = (
            logistic_predictions(experiment_c_distance_model, distance_test)
        )
        coefficient_table(
            experiment_c_distance_model, CENTROID_DISTANCE_FEATURE_COLUMNS
        ).to_csv(experiment_c_distance_dir / "coefficients.csv", index=False)
        pd.DataFrame(
            {
                "example_id": test_features["example_id"],
                "label": test_labels,
                **distance_test.to_dict(orient="list"),
                "prediction": experiment_c_distance_predictions,
                "positive_probability": experiment_c_distance_probabilities,
            }
        ).to_csv(experiment_c_distance_dir / "predictions.csv", index=False)
        experiment_predictions.setdefault("experiment_c_distance_logreg", {})[task] = (
            experiment_c_distance_predictions
        )

        combined_train = pd.concat(
            [train_base_matrix.reset_index(drop=True), distance_train.reset_index(drop=True)],
            axis=1,
        )
        combined_test = pd.concat(
            [test_base_matrix.reset_index(drop=True), distance_test.reset_index(drop=True)],
            axis=1,
        )
        experiment_c_combined_dir = task_output_dir / "experiment_c_combined_logreg"
        experiment_c_combined_dir.mkdir(parents=True, exist_ok=True)
        experiment_c_combined_model = fit_logistic_regression(
            combined_train, train_labels
        )
        experiment_c_combined_predictions, experiment_c_combined_probabilities = (
            logistic_predictions(experiment_c_combined_model, combined_test)
        )
        coefficient_table(
            experiment_c_combined_model,
            tuple(str(column) for column in combined_train.columns),
        ).to_csv(experiment_c_combined_dir / "coefficients.csv", index=False)
        pd.DataFrame(
            {
                "example_id": test_features["example_id"],
                "label": test_labels,
                "prediction": experiment_c_combined_predictions,
                "positive_probability": experiment_c_combined_probabilities,
            }
        ).to_csv(experiment_c_combined_dir / "predictions.csv", index=False)
        experiment_predictions.setdefault("experiment_c_combined_logreg", {})[task] = (
            experiment_c_combined_predictions
        )

    comparison_rows: list[dict[str, Any]] = []
    for experiment_name in EXPERIMENT_ORDER:
        evaluation = evaluate_multitask_predictions(
            gold_frame=test_bundle.gold,
            sentiment_predictions=experiment_predictions[experiment_name]["sentiment"],
            sarcasm_predictions=experiment_predictions[experiment_name]["sarcasm"],
        )
        experiment_dir = output_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        evaluation["answer_frame"].to_csv(experiment_dir / "answer.csv", index=False)
        metrics_payload = {
            "experiment": experiment_name,
            "task_metrics": evaluation["task_metrics"],
            "component_scores": evaluation["component_scores"],
            "final_score": evaluation["final_score"],
        }
        (experiment_dir / "metrics.json").write_text(
            json.dumps(metrics_payload, indent=2) + "\n"
        )
        experiment_results[experiment_name] = metrics_payload

        comparison_rows.append(
            {
                "experiment": experiment_name,
                "final_score": evaluation["final_score"],
                "sentiment_macro_f1": evaluation["task_metrics"]["sentiment"][
                    "macro_f1"
                ],
                "sarcasm_macro_f1": evaluation["task_metrics"]["sarcasm"]["macro_f1"],
                "sentiment_accuracy": evaluation["task_metrics"]["sentiment"][
                    "accuracy"
                ],
                "sarcasm_accuracy": evaluation["task_metrics"]["sarcasm"]["accuracy"],
                **evaluation["component_scores"],
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        by=["final_score", "experiment"], ascending=[False, True]
    )
    comparison.to_csv(output_dir / "experiment_comparison.csv", index=False)
    summary_payload = {
        "base_feature_columns": list(BASE_FEATURE_COLUMNS),
        "centroid_distance_feature_columns": list(CENTROID_DISTANCE_FEATURE_COLUMNS),
        "experiments": experiment_results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n")
    return {
        "train_features": train_features,
        "test_features": test_features,
        "comparison": comparison,
        "summary": summary_payload,
    }
