import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from functional_stability.baseline import evaluate_mean_baseline, load_split, run_baselines
from functional_stability.baseline_cli import main as baseline_main


class BaselineTests(unittest.TestCase):
    def test_load_split_flattens_adjacency_matrices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.jsonl"
            _write_records(path, [("a", 0.5), ("b", 0.7)])

            split = load_split(path)

        self.assertEqual(split.features.shape, (2, 4))
        self.assertEqual(split.labels.tolist(), [0.5, 0.7])
        self.assertEqual(split.sample_ids, ["a", "b"])

    def test_mean_baseline_metrics_are_computed(self):
        with tempfile.TemporaryDirectory() as directory:
            train_path = Path(directory) / "train.jsonl"
            test_path = Path(directory) / "test.jsonl"
            _write_records(train_path, [("a", 0.5), ("b", 0.7)])
            _write_records(test_path, [("c", 0.6), ("d", 0.8)])

            train = load_split(train_path)
            test = load_split(test_path)
            metrics = evaluate_mean_baseline(train, test)

        self.assertAlmostEqual(metrics.mae, 0.1)
        self.assertAlmostEqual(metrics.rmse, (0.02) ** 0.5)

    def test_run_baselines_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            test = root / "test.jsonl"
            output = root / "baseline.json"
            _write_records(train, [("a", 0.5), ("b", 0.7), ("c", 0.9)])
            _write_records(validation, [("d", 0.6)])
            _write_records(test, [("e", 0.8)])

            result = run_baselines(train, validation, test, output, ridge_alpha=0.1)
            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertIn("mean", result["models"])
        self.assertIn("ridge", result["models"])
        self.assertEqual(saved["sample_counts"]["train"], 3)

    def test_baseline_cli_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            test = root / "test.jsonl"
            output = root / "baseline.json"
            _write_records(train, [("a", 0.5), ("b", 0.7), ("c", 0.9)])
            _write_records(validation, [("d", 0.6)])
            _write_records(test, [("e", 0.8)])

            with patch("builtins.print"):
                exit_code = baseline_main(
                    [
                        "--train",
                        str(train),
                        "--validation",
                        str(validation),
                        "--test",
                        str(test),
                        "--output",
                        str(output),
                    ]
                )

        self.assertEqual(exit_code, 0)


def _write_records(path: Path, rows: list[tuple[str, float]]) -> None:
    records = []
    for sample_id, label in rows:
        records.append(
            {
                "sample_id": sample_id,
                "source": "unit-test",
                "node_count": 2,
                "edge_count": 1,
                "label": label,
                "label_method": "exact_all_terminal",
                "adjacency": [[0.0, 1.0], [1.0, 0.0]],
            }
        )
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
