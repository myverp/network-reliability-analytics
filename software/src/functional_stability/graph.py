"""Graph structures used by the reliability and dataset modules."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class Edge:
    """Undirected edge with independent availability probability."""

    source: int
    target: int
    reliability: float

    def __post_init__(self) -> None:
        if self.source == self.target:
            raise ValueError("Self-loops are not supported")
        if self.source < 0 or self.target < 0:
            raise ValueError("Node indexes must be non-negative")
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("Edge reliability must be in [0, 1]")

    def normalized(self) -> "Edge":
        source, target = sorted((self.source, self.target))
        return Edge(source, target, self.reliability)


@dataclass(frozen=True)
class Graph:
    """Small undirected graph prepared for exact reliability calculations."""

    node_count: int
    edges: tuple[Edge, ...]

    def __post_init__(self) -> None:
        if self.node_count <= 0:
            raise ValueError("Graph must contain at least one node")

        normalized_edges = tuple(edge.normalized() for edge in self.edges)
        seen: set[tuple[int, int]] = set()

        for edge in normalized_edges:
            if edge.target >= self.node_count:
                raise ValueError("Edge endpoint is outside graph node range")
            key = (edge.source, edge.target)
            if key in seen:
                raise ValueError(f"Duplicate edge {key}")
            seen.add(key)

        object.__setattr__(self, "edges", normalized_edges)

    @classmethod
    def from_tuples(
        cls,
        node_count: int,
        edges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
        reliability: float,
    ) -> "Graph":
        return cls(node_count, tuple(Edge(source, target, reliability) for source, target in edges))

    def adjacency_matrix(self) -> list[list[float]]:
        """Return a symmetric matrix of edge reliability values."""

        matrix = [[0.0 for _ in range(self.node_count)] for _ in range(self.node_count)]
        for edge in self.edges:
            matrix[edge.source][edge.target] = edge.reliability
            matrix[edge.target][edge.source] = edge.reliability
        return matrix

    def neighbors(self) -> list[list[int]]:
        adjacency: list[list[int]] = [[] for _ in range(self.node_count)]
        for edge in self.edges:
            adjacency[edge.source].append(edge.target)
            adjacency[edge.target].append(edge.source)
        return adjacency

    def is_connected(self) -> bool:
        if self.node_count == 1:
            return True
        adjacency = self.neighbors()
        visited = {0}
        queue: deque[int] = deque([0])

        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return len(visited) == self.node_count

    def relabeled_subgraph(self, nodes: list[int] | tuple[int, ...]) -> "Graph":
        """Return the induced subgraph over selected nodes with compact indexes."""

        ordered_nodes = list(dict.fromkeys(nodes))
        if not ordered_nodes:
            raise ValueError("Subgraph must contain at least one node")
        if min(ordered_nodes) < 0 or max(ordered_nodes) >= self.node_count:
            raise ValueError("Subgraph node is outside graph node range")

        node_map = {old: new for new, old in enumerate(ordered_nodes)}
        selected = set(ordered_nodes)
        edges = [
            Edge(node_map[edge.source], node_map[edge.target], edge.reliability)
            for edge in self.edges
            if edge.source in selected and edge.target in selected
        ]
        return Graph(len(ordered_nodes), tuple(edges))


def complete_graph(node_count: int, reliability: float) -> Graph:
    edges = [(i, j) for i in range(node_count) for j in range(i + 1, node_count)]
    return Graph.from_tuples(node_count, edges, reliability)


def path_graph(node_count: int, reliability: float) -> Graph:
    edges = [(i, i + 1) for i in range(node_count - 1)]
    return Graph.from_tuples(node_count, edges, reliability)


def random_graph(
    node_count: int,
    edge_probability: float,
    reliability: float,
    seed: int | None = None,
) -> Graph:
    """Generate a simple undirected graph for early experiments.

    The concrete experimental topology policy is still unresolved in the thesis
    materials. This helper is only a basic model graph generator.
    """

    if not 0.0 <= edge_probability <= 1.0:
        raise ValueError("Edge probability must be in [0, 1]")

    rng = Random(seed)
    edges: list[tuple[int, int]] = []
    for i in range(node_count):
        for j in range(i + 1, node_count):
            if rng.random() <= edge_probability:
                edges.append((i, j))
    return Graph.from_tuples(node_count, edges, reliability)
