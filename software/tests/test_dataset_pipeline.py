import csv
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from functional_stability.dataset import (
    exact_label_if_feasible,
    export_jsonl,
    export_metadata_csv,
    fixed_size_adjacency_matrix,
    make_sample,
)
from functional_stability.graph import Graph, complete_graph, path_graph
from functional_stability.subgraph import bfs_subgraph, largest_connected_component
from functional_stability.synthetic import (
    barabasi_albert_graph,
    erdos_renyi_graph,
    waxman_like_graph,
)
from functional_stability.topology_loader import (
    load_caida_as_relationships,
    load_graphml,
    load_topology_zoo_gml,
)
from helpers import temporary_directory


class DatasetPipelineTests(unittest.TestCase):
    def test_synthetic_generators_create_valid_graphs(self):
        er = erdos_renyi_graph(8, 0.2, reliability=0.95, seed=1, ensure_connected=True)
        waxman = waxman_like_graph(8, reliability=0.95, seed=1)
        ba = barabasi_albert_graph(8, attach_edges=2, reliability=0.95, seed=1)

        self.assertTrue(er.is_connected())
        self.assertTrue(waxman.is_connected())
        self.assertTrue(ba.is_connected())
        self.assertEqual(ba.node_count, 8)

    def test_fixed_adjacency_matrix_pads_to_requested_size(self):
        graph = path_graph(3, 0.9)

        matrix = fixed_size_adjacency_matrix(graph, 5)

        self.assertEqual(len(matrix), 5)
        self.assertEqual(len(matrix[0]), 5)
        self.assertEqual(matrix[0][1], 0.9)
        self.assertEqual(matrix[4][4], 0.0)

    def test_exact_label_is_skipped_for_large_edge_count(self):
        graph = complete_graph(6, 0.9)

        self.assertIsNone(exact_label_if_feasible(graph, max_edges=5))
        self.assertIsNotNone(exact_label_if_feasible(path_graph(3, 0.9), max_edges=5))

    def test_disconnected_graph_gets_exact_zero_label_even_when_large(self):
        graph = Graph.from_tuples(6, [(0, 1), (2, 3), (4, 5)], reliability=0.9)

        self.assertEqual(exact_label_if_feasible(graph, max_edges=1), 0.0)

    def test_subgraph_extractors_return_compact_graphs(self):
        graph = Graph.from_tuples(6, [(0, 1), (1, 2), (3, 4)], reliability=0.9)

        largest = largest_connected_component(graph)
        bfs = bfs_subgraph(graph, start_node=0, max_nodes=2)

        self.assertEqual(largest.node_count, 3)
        self.assertEqual(bfs.node_count, 2)
        self.assertTrue(bfs.is_connected())

    def test_graphml_loader_reads_nodes_and_edges(self):
        with temporary_directory() as directory:
            path = Path(directory) / "toy.graphml"
            path.write_text(
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

            loaded = load_graphml(path, reliability=0.88)

        self.assertEqual(loaded.graph.node_count, 3)
        self.assertEqual(len(loaded.graph.edges), 2)
        self.assertEqual(loaded.node_labels, ("a", "b", "c"))

    def test_gml_loader_reads_topology_zoo_style_file(self):
        with temporary_directory() as directory:
            path = Path(directory) / "toy.gml"
            path.write_text(
                """
graph [
  node [
    id 0
    label "Kyiv"
  ]
  node [
    id 1
    label "Lviv"
  ]
  edge [
    source 0
    target 1
  ]
]
""",
                encoding="utf-8",
            )

            loaded = load_topology_zoo_gml(path, reliability=0.77)

        self.assertEqual(loaded.graph.node_count, 2)
        self.assertEqual(len(loaded.graph.edges), 1)
        self.assertEqual(loaded.node_labels, ("Kyiv", "Lviv"))

    def test_caida_loader_reads_pipe_delimited_as_relationships(self):
        with temporary_directory() as directory:
            path = Path(directory) / "as-rel.txt"
            path.write_text(
                """
# comment
1|2|-1
2|3|0|bgp
3|3|0
""",
                encoding="utf-8",
            )

            loaded = load_caida_as_relationships(path, reliability=0.66)

        self.assertEqual(loaded.graph.node_count, 3)
        self.assertEqual(len(loaded.graph.edges), 2)
        self.assertEqual(loaded.node_labels, ("1", "2", "3"))

    def test_dataset_export_writes_jsonl_and_metadata(self):
        sample = make_sample("toy-1", "unit-test", path_graph(3, 0.9), matrix_size=4, max_label_edges=5)

        with temporary_directory() as directory:
            jsonl_path = Path(directory) / "dataset.jsonl"
            csv_path = Path(directory) / "metadata.csv"
            export_jsonl([sample], jsonl_path)
            export_metadata_csv([sample], csv_path)

            json_record = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
            with csv_path.open(encoding="utf-8") as csv_file:
                csv_record = next(csv.DictReader(csv_file))

        self.assertEqual(json_record["sample_id"], "toy-1")
        self.assertEqual(json_record["node_count"], 3)
        self.assertEqual(csv_record["label_method"], "exact_all_terminal")


if __name__ == "__main__":
    unittest.main()
