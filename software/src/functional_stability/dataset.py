"""Dataset preparation and export for reliability approximation experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .graph import Graph
from .reliability import all_terminal_reliability


@dataclass(frozen=True)
class DatasetSample:
    sample_id: str
    source: str
    graph: Graph
    adjacency: list[list[float]]
    label: float | None
    label_method: str


def fixed_size_adjacency_matrix(graph: Graph, size: int) -> list[list[float]]:
    """Return graph adjacency padded to a fixed square matrix size."""

    if size <= 0:
        raise ValueError("Matrix size must be positive")
    if graph.node_count > size:
        raise ValueError("Graph has more nodes than requested matrix size")

    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    base = graph.adjacency_matrix()
    for row_index, row in enumerate(base):
        for column_index, value in enumerate(row):
            matrix[row_index][column_index] = value
    return matrix


def exact_label_if_feasible(graph: Graph, max_edges: int = 20) -> float | None:
    """Compute exact all-terminal reliability only below an edge-count limit."""

    if max_edges < 0:
        raise ValueError("max_edges must be non-negative")
    if not graph.is_connected():
        return 0.0
    if len(graph.edges) > max_edges:
        return None
    return all_terminal_reliability(graph)


def make_sample(
    sample_id: str,
    source: str,
    graph: Graph,
    matrix_size: int,
    max_label_edges: int = 20,
) -> DatasetSample:
    label = exact_label_if_feasible(graph, max_edges=max_label_edges)
    return DatasetSample(
        sample_id=sample_id,
        source=source,
        graph=graph,
        adjacency=fixed_size_adjacency_matrix(graph, matrix_size),
        label=label,
        label_method="exact_all_terminal" if label is not None else "skipped_too_many_edges",
    )


def export_jsonl(samples: list[DatasetSample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for sample in samples:
            file.write(json.dumps(_sample_record(sample), ensure_ascii=False) + "\n")


def export_metadata_csv(samples: list[DatasetSample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id",
                "source",
                "node_count",
                "edge_count",
                "label",
                "label_method",
            ],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "source": sample.source,
                    "node_count": sample.graph.node_count,
                    "edge_count": len(sample.graph.edges),
                    "label": "" if sample.label is None else sample.label,
                    "label_method": sample.label_method,
                }
            )


def _sample_record(sample: DatasetSample) -> dict[str, object]:
    return {
        "sample_id": sample.sample_id,
        "source": sample.source,
        "node_count": sample.graph.node_count,
        "edge_count": len(sample.graph.edges),
        "label": sample.label,
        "label_method": sample.label_method,
        "adjacency": sample.adjacency,
    }
