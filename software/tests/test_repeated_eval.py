import json
import unittest
from pathlib import Path

from functional_stability.cnn_smoke import torch_available
from functional_stability.repeated_eval import run_repeated_evaluation
from helpers import temporary_directory


class RepeatedEvaluationTests(unittest.TestCase):
    @unittest.skipUnless(torch_available(), "PyTorch is not installed")
    def test_repeated_evaluation_writes_summary(self):
        with temporary_directory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            test = root / "test.jsonl"
            output = root / "report.json"
            _write_split(train, ["syn-train-1", "real-train-1"])
            _write_split(validation, ["syn-validation-1", "real-validation-1"])
            _write_split(test, ["syn-test-1", "real-test-1"])

            result = run_repeated_evaluation(
                train,
                validation,
                test,
                output,
                seeds=[42],
                epochs=1,
                batch_size=1,
                early_stopping_patience=1,
            )

        self.assertTrue(output.exists())
        self.assertIn("stronger_cnn", result["summary"])
        self.assertIn("ridge", result["summary"])
        self.assertIn("synthetic", result["summary"]["stronger_cnn"]["by_family"]["test"])
        self.assertIn("real", result["summary"]["stronger_cnn"]["by_family"]["test"])


def _write_split(path: Path, sample_ids: list[str]) -> None:
    records = [_sample_record(sample_id) for sample_id in sample_ids]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _sample_record(sample_id: str) -> dict[str, object]:
    adjacency = [[0.0 for _ in range(10)] for _ in range(10)]
    for index in range(9):
        adjacency[index][index + 1] = 0.99
        adjacency[index + 1][index] = 0.99
    source = "synthetic" if sample_id.startswith("syn-") else "graphml:test"
    return {
        "sample_id": sample_id,
        "source": source,
        "node_count": 10,
        "edge_count": 9,
        "label": 0.9,
        "label_method": "exact_all_terminal",
        "adjacency": adjacency,
    }


if __name__ == "__main__":
    unittest.main()
