import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from functional_stability.assemble_cli import main as assemble_main
from functional_stability.assembly import DatasetAssemblyError, assemble_dataset


class DatasetAssemblyTests(unittest.TestCase):
    def test_assemble_dataset_creates_deterministic_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = _write_policy(root, matrix_size=3)
            for index in range(10):
                _write_sample(root / f"sample_{index}.jsonl", f"sample-{index:02d}", matrix_size=3)

            result = assemble_dataset(
                input_glob=str(root / "*.jsonl"),
                output_dir=root / "final",
                policy_path=policy,
            )

            train_records = _read_jsonl(root / "final" / "train.jsonl")
            validation_records = _read_jsonl(root / "final" / "validation.jsonl")
            test_records = _read_jsonl(root / "final" / "test.jsonl")
            with (root / "final" / "train_metadata.csv").open(encoding="utf-8") as csv_file:
                first_metadata = next(csv.DictReader(csv_file))

        self.assertEqual(result.sample_count, 10)
        self.assertEqual(len(train_records), 7)
        self.assertEqual(len(validation_records), 1)
        self.assertEqual(len(test_records), 2)
        self.assertEqual(train_records[0]["sample_id"], "sample-00")
        self.assertEqual(first_metadata["sample_id"], "sample-00")

    def test_duplicate_sample_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = _write_policy(root, matrix_size=3)
            _write_sample(root / "a.jsonl", "duplicate", matrix_size=3)
            _write_sample(root / "b.jsonl", "duplicate", matrix_size=3)

            with self.assertRaisesRegex(DatasetAssemblyError, "Duplicate sample_id"):
                assemble_dataset(str(root / "*.jsonl"), root / "final", policy)

    def test_missing_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = _write_policy(root, matrix_size=3)
            _write_sample(root / "a.jsonl", "missing-label", matrix_size=3, label=None)

            with self.assertRaisesRegex(DatasetAssemblyError, "Missing label"):
                assemble_dataset(str(root / "*.jsonl"), root / "final", policy)

    def test_skipped_label_is_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = _write_policy(root, matrix_size=3)
            _write_sample(root / "a.jsonl", "skipped", matrix_size=3, label_method="skipped_too_many_edges")

            with self.assertRaisesRegex(DatasetAssemblyError, "Skipped or unsupported label"):
                assemble_dataset(str(root / "*.jsonl"), root / "final", policy)

    def test_inconsistent_matrix_size_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = _write_policy(root, matrix_size=4)
            _write_sample(root / "a.jsonl", "bad-matrix", matrix_size=3)

            with self.assertRaisesRegex(DatasetAssemblyError, "Inconsistent matrix row count"):
                assemble_dataset(str(root / "*.jsonl"), root / "final", policy)

    def test_assemble_cli_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = _write_policy(root, matrix_size=3)
            for index in range(4):
                _write_sample(root / f"sample_{index}.jsonl", f"cli-{index:02d}", matrix_size=3)

            with patch("builtins.print"):
                exit_code = assemble_main(
                    [
                        "--input-glob",
                        str(root / "*.jsonl"),
                        "--policy",
                        str(policy),
                        "--output-dir",
                        str(root / "final"),
                    ]
                )

        self.assertEqual(exit_code, 0)


def _write_policy(root: Path, matrix_size: int) -> Path:
    path = root / "policy.json"
    path.write_text(
        json.dumps(
            {
                "matrix_size": matrix_size,
                "split": {"train": 0.7, "validation": 0.15, "test": 0.15},
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_sample(
    path: Path,
    sample_id: str,
    matrix_size: int,
    label: float | None = 0.9,
    label_method: str = "exact_all_terminal",
) -> None:
    record = {
        "sample_id": sample_id,
        "source": "unit-test",
        "node_count": matrix_size,
        "edge_count": matrix_size - 1,
        "label": label,
        "label_method": label_method,
        "adjacency": [[0.0 for _ in range(matrix_size)] for _ in range(matrix_size)],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
