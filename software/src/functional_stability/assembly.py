"""Aggregate generated dataset samples and create deterministic splits."""

from __future__ import annotations

import csv
import glob
import json
from dataclasses import dataclass
from pathlib import Path


class DatasetAssemblyError(ValueError):
    """Raised when generated samples do not satisfy the dataset policy."""


@dataclass(frozen=True)
class SplitResult:
    sample_count: int
    train_count: int
    validation_count: int
    test_count: int
    matrix_size: int


def assemble_dataset(
    input_glob: str,
    output_dir: str | Path,
    policy_path: str | Path,
    allow_skipped_labels: bool = False,
) -> SplitResult:
    policy = _load_policy(policy_path)
    records = _load_jsonl_records(input_glob)
    _validate_records(records, policy, allow_skipped_labels=allow_skipped_labels)

    sorted_records = sorted(records, key=lambda record: record["sample_id"])
    splits = _split_records(sorted_records, policy["split"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_records in splits.items():
        _write_jsonl(split_records, output_dir / f"{split_name}.jsonl")
        _write_metadata(split_records, output_dir / f"{split_name}_metadata.csv")

    report = {
        "sample_count": len(sorted_records),
        "matrix_size": policy["matrix_size"],
        "split_counts": {name: len(items) for name, items in splits.items()},
        "source_glob": input_glob,
        "policy": str(policy_path),
    }
    (output_dir / "assembly_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return SplitResult(
        sample_count=len(sorted_records),
        train_count=len(splits["train"]),
        validation_count=len(splits["validation"]),
        test_count=len(splits["test"]),
        matrix_size=policy["matrix_size"],
    )


def _load_policy(policy_path: str | Path) -> dict[str, object]:
    policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    required = {"matrix_size", "split"}
    missing = sorted(required - set(policy))
    if missing:
        raise DatasetAssemblyError(f"Policy is missing required fields: {', '.join(missing)}")
    return policy


def _load_jsonl_records(input_glob: str) -> list[dict[str, object]]:
    paths = [Path(path) for path in sorted(glob.glob(input_glob))]
    if not paths:
        raise DatasetAssemblyError(f"No JSONL files matched input glob: {input_glob}")

    records: list[dict[str, object]] = []
    for path in paths:
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if line.strip():
                    record = json.loads(line)
                    record["_source_file"] = str(path)
                    record["_source_line"] = line_number
                    records.append(record)

    if not records:
        raise DatasetAssemblyError("Input JSONL files did not contain any records")
    return records


def _validate_records(
    records: list[dict[str, object]],
    policy: dict[str, object],
    allow_skipped_labels: bool,
) -> None:
    matrix_size = int(policy["matrix_size"])
    seen_ids: set[str] = set()

    for record in records:
        sample_id = _required_string(record, "sample_id")
        if sample_id in seen_ids:
            raise DatasetAssemblyError(f"Duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)

        label = record.get("label")
        label_method = _required_string(record, "label_method")
        if label is None:
            raise DatasetAssemblyError(f"Missing label for sample_id: {sample_id}")
        if label_method != "exact_all_terminal" and not allow_skipped_labels:
            raise DatasetAssemblyError(
                f"Skipped or unsupported label for sample_id {sample_id}: {label_method}"
            )
        if not isinstance(label, int | float):
            raise DatasetAssemblyError(f"Label must be numeric for sample_id: {sample_id}")

        adjacency = record.get("adjacency")
        if not isinstance(adjacency, list) or len(adjacency) != matrix_size:
            raise DatasetAssemblyError(
                f"Inconsistent matrix row count for sample_id {sample_id}: expected {matrix_size}"
            )
        for row in adjacency:
            if not isinstance(row, list) or len(row) != matrix_size:
                raise DatasetAssemblyError(
                    f"Inconsistent matrix column count for sample_id {sample_id}: expected {matrix_size}"
                )

        node_count = record.get("node_count")
        edge_count = record.get("edge_count")
        if not isinstance(node_count, int) or node_count <= 0:
            raise DatasetAssemblyError(f"Invalid node_count for sample_id: {sample_id}")
        if not isinstance(edge_count, int) or edge_count < 0:
            raise DatasetAssemblyError(f"Invalid edge_count for sample_id: {sample_id}")
        if node_count > matrix_size:
            raise DatasetAssemblyError(
                f"node_count exceeds matrix_size for sample_id {sample_id}: {node_count} > {matrix_size}"
            )


def _split_records(
    records: list[dict[str, object]],
    split_policy: object,
) -> dict[str, list[dict[str, object]]]:
    if not isinstance(split_policy, dict):
        raise DatasetAssemblyError("Policy split must be an object")

    train_ratio = float(split_policy["train"])
    validation_ratio = float(split_policy["validation"])
    test_ratio = float(split_policy["test"])
    ratio_sum = train_ratio + validation_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise DatasetAssemblyError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    sample_count = len(records)
    train_count = int(sample_count * train_ratio)
    validation_count = int(sample_count * validation_ratio)
    test_count = sample_count - train_count - validation_count

    return {
        "train": records[:train_count],
        "validation": records[train_count : train_count + validation_count],
        "test": records[train_count + validation_count : train_count + validation_count + test_count],
    }


def _write_jsonl(records: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            clean_record = {key: value for key, value in record.items() if not key.startswith("_")}
            file.write(json.dumps(clean_record, ensure_ascii=False) + "\n")


def _write_metadata(records: list[dict[str, object]], path: Path) -> None:
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
        for record in records:
            writer.writerow(
                {
                    "sample_id": record["sample_id"],
                    "source": record.get("source", ""),
                    "node_count": record["node_count"],
                    "edge_count": record["edge_count"],
                    "label": record["label"],
                    "label_method": record["label_method"],
                }
            )


def _required_string(record: dict[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise DatasetAssemblyError(f"Missing or invalid string field: {key}")
    return value
