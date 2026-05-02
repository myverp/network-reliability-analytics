"""Simple non-CNN baselines for reliability prediction."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SplitArrays:
    features: np.ndarray
    labels: np.ndarray
    sample_ids: list[str]


@dataclass(frozen=True)
class BaselineMetrics:
    mae: float
    rmse: float
    inference_seconds: float


def load_split(path: str | Path) -> SplitArrays:
    records = _read_jsonl(path)
    if not records:
        raise ValueError(f"Split file is empty: {path}")

    features = []
    labels = []
    sample_ids = []
    for record in records:
        adjacency = np.array(record["adjacency"], dtype=float)
        features.append(adjacency.reshape(-1))
        labels.append(float(record["label"]))
        sample_ids.append(str(record["sample_id"]))

    return SplitArrays(
        features=np.vstack(features),
        labels=np.array(labels, dtype=float),
        sample_ids=sample_ids,
    )


def evaluate_mean_baseline(train: SplitArrays, evaluation: SplitArrays) -> BaselineMetrics:
    prediction_value = float(np.mean(train.labels))
    start = time.perf_counter()
    predictions = np.full_like(evaluation.labels, prediction_value, dtype=float)
    elapsed = time.perf_counter() - start
    return _metrics(evaluation.labels, predictions, elapsed)


def evaluate_ridge_baseline(
    train: SplitArrays,
    evaluation: SplitArrays,
    alpha: float = 1.0,
) -> BaselineMetrics:
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    train_features = _with_bias(train.features)
    eval_features = _with_bias(evaluation.features)
    regularization = alpha * np.eye(train_features.shape[1])
    regularization[-1, -1] = 0.0

    weights = np.linalg.solve(
        train_features.T @ train_features + regularization,
        train_features.T @ train.labels,
    )

    start = time.perf_counter()
    predictions = eval_features @ weights
    elapsed = time.perf_counter() - start
    return _metrics(evaluation.labels, predictions, elapsed)


def run_baselines(
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    output_path: str | Path,
    ridge_alpha: float = 1.0,
) -> dict[str, object]:
    train = load_split(train_path)
    validation = load_split(validation_path)
    test = load_split(test_path)

    result = {
        "sample_counts": {
            "train": len(train.labels),
            "validation": len(validation.labels),
            "test": len(test.labels),
        },
        "models": {
            "mean": {
                "validation": _as_dict(evaluate_mean_baseline(train, validation)),
                "test": _as_dict(evaluate_mean_baseline(train, test)),
            },
            "ridge": {
                "alpha": ridge_alpha,
                "validation": _as_dict(evaluate_ridge_baseline(train, validation, ridge_alpha)),
                "test": _as_dict(evaluate_ridge_baseline(train, test, ridge_alpha)),
            },
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _with_bias(features: np.ndarray) -> np.ndarray:
    return np.hstack([features, np.ones((features.shape[0], 1), dtype=float)])


def _metrics(actual: np.ndarray, predicted: np.ndarray, inference_seconds: float) -> BaselineMetrics:
    errors = predicted - actual
    return BaselineMetrics(
        mae=float(np.mean(np.abs(errors))),
        rmse=float(math.sqrt(float(np.mean(errors * errors)))),
        inference_seconds=float(inference_seconds),
    )


def _as_dict(metrics: BaselineMetrics) -> dict[str, float]:
    return {
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "inference_seconds": metrics.inference_seconds,
    }
