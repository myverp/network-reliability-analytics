import csv
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from functional_stability.cli import main
from helpers import temporary_directory


class CliTests(unittest.TestCase):
    def test_cli_exports_synthetic_dataset(self):
        with temporary_directory() as directory:
            output_jsonl = Path(directory) / "dataset.jsonl"
            output_csv = Path(directory) / "metadata.csv"

            with patch("builtins.print"):
                exit_code = main(
                    [
                        "--sample-id",
                        "cli-synthetic",
                        "--source",
                        "synthetic",
                        "--synthetic-model",
                        "erdos-renyi",
                        "--nodes",
                        "5",
                        "--edge-probability",
                        "0.3",
                        "--edge-reliability",
                        "0.9",
                        "--seed",
                        "7",
                        "--matrix-size",
                        "5",
                        "--max-label-edges",
                        "20",
                        "--output-jsonl",
                        str(output_jsonl),
                        "--output-metadata",
                        str(output_csv),
                    ]
                )

            record = json.loads(output_jsonl.read_text(encoding="utf-8").splitlines()[0])
            with output_csv.open(encoding="utf-8") as csv_file:
                metadata = next(csv.DictReader(csv_file))

        self.assertEqual(exit_code, 0)
        self.assertEqual(record["sample_id"], "cli-synthetic")
        self.assertEqual(metadata["source"], "synthetic:erdos-renyi")

    def test_cli_exports_file_dataset_with_bfs_subgraph(self):
        with temporary_directory() as directory:
            topology = Path(directory) / "toy.graphml"
            output_jsonl = Path(directory) / "dataset.jsonl"
            output_csv = Path(directory) / "metadata.csv"
            topology.write_text(
                """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <graph edgedefault="undirected">
    <node id="a"/>
    <node id="b"/>
    <node id="c"/>
    <edge source="a" target="b"/>
    <edge source="b" target="c"/>
  </graph>
</graphml>
""",
                encoding="utf-8",
            )

            with patch("builtins.print"):
                exit_code = main(
                    [
                        "--sample-id",
                        "cli-file",
                        "--source",
                        "file",
                        "--topology-file",
                        str(topology),
                        "--topology-type",
                        "graphml",
                        "--edge-reliability",
                        "0.95",
                        "--subgraph-method",
                        "bfs",
                        "--subgraph-nodes",
                        "2",
                        "--matrix-size",
                        "2",
                        "--output-jsonl",
                        str(output_jsonl),
                        "--output-metadata",
                        str(output_csv),
                    ]
                )

            record = json.loads(output_jsonl.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(exit_code, 0)
        self.assertEqual(record["node_count"], 2)
        self.assertEqual(record["label_method"], "exact_all_terminal")


if __name__ == "__main__":
    unittest.main()
