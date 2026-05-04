"""Minimal PyTorch CNN smoke experiment for reliability regression."""

from __future__ import annotations

import importlib.util
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


RIDGE_TEST_MAE_REFERENCE = 0.035949


@dataclass(frozen=True)
class CnnSplit:
    features: np.ndarray
    labels: np.ndarray
    sample_ids: list[str]


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def load_cnn_split(path: str | Path, matrix_size: int = 10) -> CnnSplit:
    records = _read_jsonl(path)
    if not records:
        raise ValueError(f"Split file is empty: {path}")

    features: list[np.ndarray] = []
    labels: list[float] = []
    sample_ids: list[str] = []
    expected_shape = (matrix_size, matrix_size)

    for record in records:
        adjacency = np.array(record["adjacency"], dtype=np.float32)
        if adjacency.shape != expected_shape:
            raise ValueError(
                f"Sample {record.get('sample_id')} has adjacency shape "
                f"{adjacency.shape}, expected {expected_shape}"
            )
        if record.get("label") is None:
            raise ValueError(f"Sample {record.get('sample_id')} is missing label")

        features.append(adjacency)
        labels.append(float(record["label"]))
        sample_ids.append(str(record["sample_id"]))

    return CnnSplit(
        features=np.stack(features).reshape((-1, 1, matrix_size, matrix_size)),
        labels=np.array(labels, dtype=np.float32).reshape((-1, 1)),
        sample_ids=sample_ids,
    )


def run_cnn_smoke(
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    output_path: str | Path,
    matrix_size: int = 10,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    seed: int = 42,
    early_stopping_patience: int = 20,
    ridge_test_mae_reference: float = RIDGE_TEST_MAE_REFERENCE,
) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")

    torch, nn, DataLoader, TensorDataset = _require_torch()
    _seed_everything(torch, seed)

    train = load_cnn_split(train_path, matrix_size=matrix_size)
    validation = load_cnn_split(validation_path, matrix_size=matrix_size)
    test = load_cnn_split(test_path, matrix_size=matrix_size)
    train, validation, test, normalization = _normalize_splits(train, validation, test)

    candidates = {
        "small_cnn": {
            "description": "Conv2d(1,8,3,pad=1) -> ReLU -> Conv2d(8,16,3,pad=1) -> ReLU -> AdaptiveAvgPool2d(1,1) -> Linear(16,16) -> ReLU -> Linear(16,1)",
            "factory": lambda: _build_small_cnn(nn),
        },
        "stronger_cnn": {
            "description": "Conv2d(1,16,3,pad=1) -> ReLU -> Conv2d(16,32,3,pad=1) -> ReLU -> AdaptiveAvgPool2d(2,2) -> Linear(128,32) -> ReLU -> Linear(32,1)",
            "factory": lambda: _build_stronger_cnn(nn),
        },
        "mlp": {
            "description": "Flatten(10x10) -> Linear(100,64) -> ReLU -> Linear(64,32) -> ReLU -> Linear(32,1)",
            "factory": lambda: _build_mlp(nn, matrix_size),
        },
    }

    model_results = {}
    for model_index, (model_name, candidate) in enumerate(candidates.items()):
        _seed_everything(torch, seed + model_index)
        model_results[model_name] = _train_one_model(
            model=candidate["factory"](),
            description=str(candidate["description"]),
            train=train,
            validation=validation,
            test=test,
            torch=torch,
            nn=nn,
            DataLoader=DataLoader,
            TensorDataset=TensorDataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + model_index,
            early_stopping_patience=early_stopping_patience,
        )

    best_model_name = min(
        model_results,
        key=lambda name: model_results[name]["metrics"]["validation"]["mae"],
    )
    best_model = model_results[best_model_name]

    result = {
        "experiment": "cnn_smoke",
        "input_shape": [1, matrix_size, matrix_size],
        "output_shape": [1],
        "normalization": normalization,
        "best_model": best_model_name,
        "architecture": best_model["architecture"],
        "training": {
            "max_epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "early_stopping": {
                "monitor": "validation_mae",
                "patience": early_stopping_patience,
            },
        },
        "sample_counts": {
            "train": len(train.labels),
            "validation": len(validation.labels),
            "test": len(test.labels),
        },
        "metrics": best_model["metrics"],
        "baseline_comparison": {
            "ridge_test_mae_reference": ridge_test_mae_reference,
            "best_model_test_mae": best_model["metrics"]["test"]["mae"],
            "beats_ridge_test_mae": best_model["metrics"]["test"]["mae"] < ridge_test_mae_reference,
        },
        "models": model_results,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_stronger_cnn_multiseed(
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
    ridge_test_mae_reference: float = RIDGE_TEST_MAE_REFERENCE,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("at least one seed is required")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")

    torch, nn, DataLoader, TensorDataset = _require_torch()
    train = load_cnn_split(train_path, matrix_size=matrix_size)
    validation = load_cnn_split(validation_path, matrix_size=matrix_size)
    test = load_cnn_split(test_path, matrix_size=matrix_size)
    train, validation, test, normalization = _normalize_splits(train, validation, test)

    per_seed = []
    for seed in seeds:
        _seed_everything(torch, seed)
        model_result = _train_one_model(
            model=_build_stronger_cnn(nn),
            description="Conv2d(1,16,3,pad=1) -> ReLU -> Conv2d(16,32,3,pad=1) -> ReLU -> AdaptiveAvgPool2d(2,2) -> Linear(128,32) -> ReLU -> Linear(32,1)",
            train=train,
            validation=validation,
            test=test,
            torch=torch,
            nn=nn,
            DataLoader=DataLoader,
            TensorDataset=TensorDataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            early_stopping_patience=early_stopping_patience,
        )
        per_seed.append(
            {
                "seed": seed,
                "metrics": model_result["metrics"],
                "training": model_result["training"],
                "beats_ridge_test_mae": model_result["metrics"]["test"]["mae"]
                < ridge_test_mae_reference,
            }
        )

    result = {
        "experiment": "stronger_cnn_multiseed",
        "model": "stronger_cnn",
        "input_shape": [1, matrix_size, matrix_size],
        "output_shape": [1],
        "architecture": {
            "description": "Conv2d(1,16,3,pad=1) -> ReLU -> Conv2d(16,32,3,pad=1) -> ReLU -> AdaptiveAvgPool2d(2,2) -> Linear(128,32) -> ReLU -> Linear(32,1)",
            "parameter_count": _parameter_count(_build_stronger_cnn(nn)),
        },
        "normalization": normalization,
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "early_stopping": {
                "monitor": "validation_mae",
                "patience": early_stopping_patience,
            },
            "seeds": seeds,
        },
        "sample_counts": {
            "train": len(train.labels),
            "validation": len(validation.labels),
            "test": len(test.labels),
        },
        "ridge_test_mae_reference": ridge_test_mae_reference,
        "per_seed": per_seed,
        "summary": _summarize_multiseed(per_seed),
        "consistently_beats_ridge_test_mae": all(
            row["beats_ridge_test_mae"] for row in per_seed
        ),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _require_torch() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyTorch is required for the CNN smoke experiment. "
            "Install it with: python -m pip install torch"
        ) from error
    return torch, nn, DataLoader, TensorDataset


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _normalize_splits(
    train: CnnSplit,
    validation: CnnSplit,
    test: CnnSplit,
) -> tuple[CnnSplit, CnnSplit, CnnSplit, dict[str, float | str]]:
    mean = float(np.mean(train.features))
    std = float(np.std(train.features))
    if std == 0.0:
        std = 1.0

    def normalize(split: CnnSplit) -> CnnSplit:
        return CnnSplit(
            features=((split.features - mean) / std).astype(np.float32),
            labels=split.labels,
            sample_ids=split.sample_ids,
        )

    return (
        normalize(train),
        normalize(validation),
        normalize(test),
        {
            "method": "global_train_standardization",
            "mean": mean,
            "std": std,
        },
    )


def _train_one_model(
    model: Any,
    description: str,
    train: CnnSplit,
    validation: CnnSplit,
    test: CnnSplit,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    TensorDataset: Any,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    early_stopping_patience: int,
    return_model: bool = False,
) -> dict[str, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train.features),
            torch.from_numpy(train.labels),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    start = time.perf_counter()
    history = []
    best_validation_mae = math.inf
    best_epoch = 0
    best_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = loss_function(predictions, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(batch_labels)
            total_count += len(batch_labels)

        validation_metrics = _evaluate(model, validation, torch)
        validation_mae = validation_metrics["mae"]
        history.append(
            {
                "epoch": epoch,
                "train_mse": total_loss / total_count,
                "validation_mae": validation_mae,
                "validation_rmse": validation_metrics["rmse"],
            }
        )

        if validation_mae < best_validation_mae:
            best_validation_mae = validation_mae
            best_epoch = epoch
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                break

    model.load_state_dict(best_state)
    training_seconds = time.perf_counter() - start
    validation_metrics = _evaluate(model, validation, torch)
    test_metrics = _evaluate(model, test, torch)

    result = {
        "architecture": {
            "description": description,
            "parameter_count": _parameter_count(model),
        },
        "training": {
            "seed": seed,
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "stopped_early": len(history) < epochs,
            "training_seconds": training_seconds,
        },
        "metrics": {
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "history": history,
    }
    if return_model:
        result["model"] = model
    return result


def _build_small_cnn(nn: Any) -> Any:
    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )


def _build_stronger_cnn(nn: Any) -> Any:
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((2, 2)),
        nn.Flatten(),
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


def _build_mlp(nn: Any, matrix_size: int) -> Any:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(matrix_size * matrix_size, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


def _evaluate(model: Any, split: CnnSplit, torch: Any) -> dict[str, float]:
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(split.features)).numpy()
    inference_seconds = time.perf_counter() - start
    errors = predictions.reshape(-1) - split.labels.reshape(-1)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(math.sqrt(float(np.mean(errors * errors)))),
        "inference_seconds": float(inference_seconds),
    }


def _summarize_multiseed(per_seed: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for split_name in ("validation", "test"):
        summary[split_name] = {}
        for metric_name in ("mae", "rmse"):
            values = np.array(
                [
                    row["metrics"][split_name][metric_name]
                    for row in per_seed
                ],
                dtype=float,
            )
            summary[split_name][metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
            }
    return summary


def _parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
