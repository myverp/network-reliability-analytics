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
    ridge_test_mae_reference: float = RIDGE_TEST_MAE_REFERENCE,
) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    torch, nn, DataLoader, TensorDataset = _require_torch()
    _seed_everything(torch, seed)

    train = load_cnn_split(train_path, matrix_size=matrix_size)
    validation = load_cnn_split(validation_path, matrix_size=matrix_size)
    test = load_cnn_split(test_path, matrix_size=matrix_size)

    model = _build_model(nn)
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
        history.append(
            {
                "epoch": epoch,
                "train_mse": total_loss / total_count,
                "validation_mae": validation_metrics["mae"],
                "validation_rmse": validation_metrics["rmse"],
            }
        )

    training_seconds = time.perf_counter() - start
    validation_metrics = _evaluate(model, validation, torch)
    test_metrics = _evaluate(model, test, torch)

    result = {
        "experiment": "cnn_smoke",
        "input_shape": [1, matrix_size, matrix_size],
        "output_shape": [1],
        "architecture": {
            "description": "Conv2d(1,8,3,pad=1) -> ReLU -> Conv2d(8,16,3,pad=1) -> ReLU -> AdaptiveAvgPool2d(1,1) -> Linear(16,16) -> ReLU -> Linear(16,1)",
            "parameter_count": _parameter_count(model),
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "training_seconds": training_seconds,
        },
        "sample_counts": {
            "train": len(train.labels),
            "validation": len(validation.labels),
            "test": len(test.labels),
        },
        "metrics": {
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "baseline_comparison": {
            "ridge_test_mae_reference": ridge_test_mae_reference,
            "cnn_test_mae": test_metrics["mae"],
            "beats_ridge_test_mae": test_metrics["mae"] < ridge_test_mae_reference,
        },
        "history": history,
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


def _build_model(nn: Any) -> Any:
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


def _parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
