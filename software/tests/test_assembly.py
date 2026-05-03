import csv
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from functional_stability.assemble_cli import main as assemble_main
from functional_stability.assembly import DatasetAssemblyError, assemble_dataset
from helpers import temporary_directory


class DatasetAssemblyTests(unittest.TestCase):
    def test_assemble_dataset_creates_deterministic_splits(self):
        with temporary_directory() as directory:
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
        self.assertEqual(len({record["sample_id"] for record in train_records}), 7)
        self.assertIn(first_metadata["sample_id"], {record["sample_id"] for record in train_records})

    def test_split_is_stratified_by_sample_family(self):
        with temporary_directory() as directory:
            root = Path(directory)
            policy = _write_policy(root, matrix_size=3)
            for index in range(10):
                _write_sample(root / f"er_{index}.jsonl", f"syn-erdos-renyi-n6-r0.99-{index}", matrix_size=3)
                _write_sample(root / f"wx_{index}.jsonl", f"syn-waxman-n6-r0.99-{index}", matrix_size=3)

            assemble_dataset(str(root / "*.jsonl"), root / "final", policy)

            train_records = _read_jsonl(root / "final" / "train.jsonl")
            validation_records = _read_jsonl(root / "final" / "validation.jsonl")
            test_records = _read_jsonl(root / "final" / "test.jsonl")

        self.assertEqual(len(train_records), 14)
        self.assertEqual(len(validation_records), 2)
        self.assertEqual(len(test_records), 4)
        self.assertEqual(_count_prefix(train_records, "syn-erdos-renyi"), 7)
        self.assertEqual(_count_prefix(train_records, "syn-waxman"), 7)
        self.assertEqual(_count_prefix(test_records, "syn-erdos-renyi"), 2)
        self.assertEqual(_count_prefix(test_records, "syn-waxman"), 2)

    def test_duplicate_sample_id_is_rejected(self):
        with temporary_directory() as directory:
            root = Path(directory)
            policy = _write_policy(root, matrix_size=3)
            _write_sample(root / "a.jsonl", "duplicate", matrix_size=3)
            _write_sample(root / "b.jsonl", "duplicate", matrix_size=3)

            with self.assertRaisesRegex(DatasetAssemblyError, "Duplicate sample_id"):
                assemble_dataset(str(root / "*.jsonl"), root / "final", policy)

    def test_missing_label_is_rejected(self):
        with temporary_directory() as directory:
            root = Path(directory)
            policy = _write_policy(root, matrix_size=3)
            _write_sample(root / "a.jsonl", "missing-label", matrix_size=3, label=None)

            with self.assertRaisesRegex(DatasetAssemblyError, "Missing label"):
                assemble_dataset(str(root / "*.jsonl"), root / "final", policy)

    def test_skipped_label_is_rejected_by_default(self):
        with temporary_directory() as directory:
            root = Path(directory)
            policy = _write_policy(root, matrix_size=3)
            _write_sample(root / "a.jsonl", "skipped", matrix_size=3, label_method="skipped_too_many_edges")

            with self.assertRaisesRegex(DatasetAssemblyError, "Skipped or unsupported label"):
                assemble_dataset(str(root / "*.jsonl"), root / "final", policy)

    def test_inconsistent_matrix_size_is_rejected(self):
        with temporary_directory() as directory:
            root = Path(directory)
            policy = _write_policy(root, matrix_size=4)
            _write_sample(root / "a.jsonl", "bad-matrix", matrix_size=3)

            with self.assertRaisesRegex(DatasetAssemblyError, "Inconsistent matrix row count"):
                assemble_dataset(str(root / "*.jsonl"), root / "final", policy)

    def test_assemble_cli_runs(self):
        with temporary_directory() as directory:
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


def _count_prefix(records: list[dict[str, object]], prefix: str) -> int:
    return sum(1 for record in records if str(record["sample_id"]).startswith(prefix))


if __name__ == "__main__":
    unittest.main()
