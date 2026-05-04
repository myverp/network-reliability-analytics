"""Repeated stronger CNN and ridge evaluation."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .baseline import load_split as load_baseline_split
from .baseline import _with_bias
from .cnn_smoke import (
    RIDGE_TEST_MAE_REFERENCE,
    _build_stronger_cnn,
    _normalize_splits,
    _parameter_count,
    _require_torch,
    _seed_everything,
    _train_one_model,
    load_cnn_split,
)


def run_repeated_evaluation(
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    output_path: str | Path,
    seeds: list[int],
    matrix_size: int = 10,
    epochs: int = 300,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 30,
    ridge_alpha: float = 1.0,
    ridge_test_mae_reference: float = RIDGE_TEST_MAE_REFERENCE,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("at least one seed is required")

    torch, nn, DataLoader, TensorDataset = _require_torch()
    cnn_train = load_cnn_split(train_path, matrix_size=matrix_size)
    cnn_validation = load_cnn_split(validation_path, matrix_size=matrix_size)
    cnn_test = load_cnn_split(test_path, matrix_size=matrix_size)
    cnn_train, cnn_validation, cnn_test, normalization = _normalize_splits(
        cnn_train,
        cnn_validation,
        cnn_test,
    )

    validation_records = _read_jsonl(validation_path)
    test_records = _read_jsonl(test_path)

    neural_runs = []
    for seed in seeds:
        _seed_everything(torch, seed)
        model_result = _train_one_model(
            model=_build_stronger_cnn(nn),
            description="stronger_cnn",
            train=cnn_train,
            validation=cnn_validation,
            test=cnn_test,
            torch=torch,
            nn=nn,
            DataLoader=DataLoader,
            TensorDataset=TensorDataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            early_stopping_patience=early_stopping_patience,
            return_model=True,
        )
        model = model_result.pop("model")
        neural_runs.append(
            {
                "seed": seed,
                "training": model_result["training"],
                "metrics": model_result["metrics"],
                "by_family": {
                    "validation": _family_metrics(
                        validation_records,
                        _predict_cnn(model, cnn_validation, torch),
                    ),
                    "test": _family_metrics(
                        test_records,
                        _predict_cnn(model, cnn_test, torch),
                    ),
                },
                "beats_ridge_test_mae_reference": model_result["metrics"]["test"]["mae"]
                < ridge_test_mae_reference,
            }
        )

    ridge = _run_ridge_evaluation(
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
        validation_records=validation_records,
        test_records=test_records,
        alpha=ridge_alpha,
    )
    summary = {
        "stronger_cnn": {
            "overall": _summarize_runs(neural_runs),
            "by_family": _summarize_family_runs(neural_runs),
        },
        "ridge": {
            "overall": _ridge_summary(ridge),
            "by_family": _ridge_family_summary(ridge),
        },
    }

    result = {
        "experiment": "repeated_stronger_cnn_vs_ridge",
        "seeds": seeds,
        "input_shape": [1, matrix_size, matrix_size],
        "output_shape": [1],
        "normalization": normalization,
        "models": {
            "stronger_cnn": {
                "architecture": {
                    "description": "Conv2d(1,16,3,pad=1) -> ReLU -> Conv2d(16,32,3,pad=1) -> ReLU -> AdaptiveAvgPool2d(2,2) -> Linear(128,32) -> ReLU -> Linear(32,1)",
                    "parameter_count": _parameter_count(_build_stronger_cnn(nn)),
                },
                "runs": neural_runs,
            },
            "ridge": ridge,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "early_stopping_patience": early_stopping_patience,
            "ridge_alpha": ridge_alpha,
        },
        "summary": summary,
        "comparison": _comparison(summary),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _run_ridge_evaluation(
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    validation_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    alpha: float,
) -> dict[str, Any]:
    train = load_baseline_split(train_path)
    validation = load_baseline_split(validation_path)
    test = load_baseline_split(test_path)

    train_features = _with_bias(train.features)
    regularization = alpha * np.eye(train_features.shape[1])
    regularization[-1, -1] = 0.0
    weights = np.linalg.solve(
        train_features.T @ train_features + regularization,
        train_features.T @ train.labels,
    )

    start = time.perf_counter()
    validation_predictions = _with_bias(validation.features) @ weights
    test_predictions = _with_bias(test.features) @ weights
    inference_seconds = time.perf_counter() - start

    return {
        "alpha": alpha,
        "metrics": {
            "validation": _metrics(validation.labels, validation_predictions, inference_seconds),
            "test": _metrics(test.labels, test_predictions, inference_seconds),
        },
        "by_family": {
            "validation": _family_metrics(validation_records, validation_predictions),
            "test": _family_metrics(test_records, test_predictions),
        },
    }


def _predict_cnn(model: Any, split: Any, torch: Any) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(split.features)).numpy().reshape(-1)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _family(record: dict[str, Any]) -> str:
    sample_id = str(record.get("sample_id", ""))
    source = str(record.get("source", ""))
    if sample_id.startswith("syn-") or source == "synthetic":
        return "synthetic"
    if sample_id.startswith("real-") or source.startswith("graphml:"):
        return "real"
    return "unknown"


def _family_metrics(records: list[dict[str, Any]], predictions: np.ndarray) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for record, prediction in zip(records, predictions):
        family = _family(record)
        grouped.setdefault(family, {"actual": [], "predicted": []})
        grouped[family]["actual"].append(float(record["label"]))
        grouped[family]["predicted"].append(float(prediction))

    result = {}
    for family, values in grouped.items():
        result[family] = _metrics(
            np.array(values["actual"], dtype=float),
            np.array(values["predicted"], dtype=float),
            inference_seconds=0.0,
        )
        result[family]["count"] = len(values["actual"])
    return result


def _metrics(actual: np.ndarray, predicted: np.ndarray, inference_seconds: float) -> dict[str, float]:
    errors = predicted - actual
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(math.sqrt(float(np.mean(errors * errors)))),
        "inference_seconds": float(inference_seconds),
    }


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        split: {
            metric: _summary_values(
                [run["metrics"][split][metric] for run in runs]
            )
            for metric in ("mae", "rmse")
        }
        for split in ("validation", "test")
    }


def _summarize_family_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    families = sorted(
        {
            family
            for run in runs
            for split in ("validation", "test")
            for family in run["by_family"][split]
        }
    )
    return {
        split: {
            family: {
                metric: _summary_values(
                    [
                        run["by_family"][split][family][metric]
                        for run in runs
                        if family in run["by_family"][split]
                    ]
                )
                for metric in ("mae", "rmse")
            }
            for family in families
        }
        for split in ("validation", "test")
    }


def _ridge_summary(ridge: dict[str, Any]) -> dict[str, Any]:
    return {
        split: {
            metric: _summary_values([ridge["metrics"][split][metric]])
            for metric in ("mae", "rmse")
        }
        for split in ("validation", "test")
    }


def _ridge_family_summary(ridge: dict[str, Any]) -> dict[str, Any]:
    return {
        split: {
            family: {
                metric: _summary_values([metrics[metric]])
                for metric in ("mae", "rmse")
            }
            for family, metrics in ridge["by_family"][split].items()
        }
        for split in ("validation", "test")
    }


def _summary_values(values: list[float]) -> dict[str, float]:
    data = np.array(values, dtype=float)
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
    ci_half_width = 1.96 * std / math.sqrt(len(data)) if len(data) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - ci_half_width,
        "ci95_high": mean + ci_half_width,
    }


def _comparison(summary: dict[str, Any]) -> dict[str, Any]:
    cnn_test_mae = summary["stronger_cnn"]["overall"]["test"]["mae"]["mean"]
    ridge_test_mae = summary["ridge"]["overall"]["test"]["mae"]["mean"]
    return {
        "test_mae_difference_cnn_minus_ridge": cnn_test_mae - ridge_test_mae,
        "cnn_mean_test_mae_beats_ridge": cnn_test_mae < ridge_test_mae,
    }
