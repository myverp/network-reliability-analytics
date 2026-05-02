import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from functional_stability.graph import Edge, Graph, complete_graph, path_graph
from functional_stability.reliability import all_terminal_reliability


class ReliabilityTests(unittest.TestCase):
    def test_single_node_graph_is_always_connected(self):
        graph = Graph(1, ())

        self.assertEqual(all_terminal_reliability(graph), 1.0)

    def test_two_node_graph_reliability_equals_edge_reliability(self):
        graph = Graph(2, (Edge(0, 1, 0.73),))

        self.assertAlmostEqual(all_terminal_reliability(graph), 0.73)

    def test_path_graph_requires_all_edges_to_work(self):
        graph = path_graph(4, 0.9)

        self.assertAlmostEqual(all_terminal_reliability(graph), 0.9**3)

    def test_triangle_reliability_matches_closed_form(self):
        graph = complete_graph(3, 0.8)
        expected = 3 * (0.8**2) * (1 - 0.8) + 0.8**3

        self.assertTrue(math.isclose(all_terminal_reliability(graph), expected, rel_tol=1e-12))

    def test_adjacency_matrix_contains_edge_reliabilities(self):
        graph = Graph(3, (Edge(0, 2, 0.6),))

        self.assertEqual(
            graph.adjacency_matrix(),
            [
                [0.0, 0.0, 0.6],
                [0.0, 0.0, 0.0],
                [0.6, 0.0, 0.0],
            ],
        )


if __name__ == "__main__":
    unittest.main()
