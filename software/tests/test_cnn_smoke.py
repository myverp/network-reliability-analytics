import json
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from functional_stability.cnn_smoke_cli import main as cnn_smoke_main
from functional_stability.cnn_smoke import (
    load_cnn_split,
    run_cnn_smoke,
    run_stronger_cnn_multiseed,
    torch_available,
)
from helpers import temporary_directory


class CnnSmokeTests(unittest.TestCase):
    def test_load_cnn_split_shapes_adjacency_for_conv2d(self):
        with temporary_directory() as directory:
            path = Path(directory) / "split.jsonl"
            _write_split(path, ["sample-1"])

            split = load_cnn_split(path)

        self.assertEqual((1, 1, 10, 10), split.features.shape)
        self.assertEqual((1, 1), split.labels.shape)
        self.assertEqual(["sample-1"], split.sample_ids)

    def test_load_cnn_split_rejects_wrong_matrix_size(self):
        with temporary_directory() as directory:
            path = Path(directory) / "split.jsonl"
            record = _sample_record("bad-shape")
            record["adjacency"] = [[0.0, 1.0], [1.0, 0.0]]
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_cnn_split(path)

    @unittest.skipIf(torch_available(), "PyTorch is installed")
    def test_run_cnn_smoke_reports_missing_torch(self):
        with temporary_directory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            test = root / "test.jsonl"
            _write_split(train, ["train-1"])
            _write_split(validation, ["validation-1"])
            _write_split(test, ["test-1"])

            with self.assertRaisesRegex(RuntimeError, "PyTorch is required"):
                run_cnn_smoke(train, validation, test, root / "report.json", epochs=1)

    @unittest.skipIf(torch_available(), "PyTorch is installed")
    def test_cnn_cli_writes_not_run_report_when_torch_is_missing(self):
        with temporary_directory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            test = root / "test.jsonl"
            output = root / "report.json"
            _write_split(train, ["train-1"])
            _write_split(validation, ["validation-1"])
            _write_split(test, ["test-1"])

            with redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    cnn_smoke_main(
                        [
                            "--train",
                            str(train),
                            "--validation",
                            str(validation),
                            "--test",
                            str(test),
                            "--output",
                            str(output),
                            "--epochs",
                            "1",
                        ]
                    )

            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual("not_run", report["status"])
        self.assertEqual("torch", report["required_dependency"])

    @unittest.skipUnless(torch_available(), "PyTorch is not installed")
    def test_run_cnn_smoke_writes_report_when_torch_is_available(self):
        with temporary_directory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            test = root / "test.jsonl"
            output = root / "report.json"
            _write_split(train, ["train-1", "train-2"])
            _write_split(validation, ["validation-1"])
            _write_split(test, ["test-1"])

            result = run_cnn_smoke(train, validation, test, output, epochs=1, batch_size=1)

        self.assertTrue(output.exists())
        self.assertIn("test", result["metrics"])
        self.assertEqual([1, 10, 10], result["input_shape"])

    @unittest.skipUnless(torch_available(), "PyTorch is not installed")
    def test_stronger_cnn_multiseed_writes_summary_when_torch_is_available(self):
        with temporary_directory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            test = root / "test.jsonl"
            output = root / "multiseed.json"
            _write_split(train, ["train-1", "train-2"])
            _write_split(validation, ["validation-1"])
            _write_split(test, ["test-1"])

            result = run_stronger_cnn_multiseed(
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
        self.assertEqual("stronger_cnn", result["model"])
        self.assertEqual(1, len(result["per_seed"]))
        self.assertIn("summary", result)


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
    return {
        "sample_id": sample_id,
        "source": "test",
        "node_count": 10,
        "edge_count": 9,
        "label": 0.9,
        "label_method": "exact_all_terminal",
        "adjacency": adjacency,
    }


if __name__ == "__main__":
    unittest.main()
