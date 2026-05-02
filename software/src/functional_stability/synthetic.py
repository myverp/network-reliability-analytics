"""Synthetic network topology generators for dataset preparation."""

from __future__ import annotations

from random import Random

from .graph import Graph


def erdos_renyi_graph(
    node_count: int,
    edge_probability: float,
    reliability: float,
    seed: int | None = None,
    ensure_connected: bool = False,
) -> Graph:
    """Generate a simple G(n, p) undirected graph."""

    if not 0.0 <= edge_probability <= 1.0:
        raise ValueError("Edge probability must be in [0, 1]")

    rng = Random(seed)
    edges: list[tuple[int, int]] = []
    for i in range(node_count):
        for j in range(i + 1, node_count):
            if rng.random() <= edge_probability:
                edges.append((i, j))

    if ensure_connected and node_count > 1:
        present = {tuple(sorted(edge)) for edge in edges}
        for node in range(1, node_count):
            edge = (rng.randrange(0, node), node)
            if edge not in present:
                edges.append(edge)
                present.add(edge)

    return Graph.from_tuples(node_count, edges, reliability)


def waxman_like_graph(
    node_count: int,
    reliability: float,
    alpha: float = 0.35,
    seed: int | None = None,
    ensure_connected: bool = True,
) -> Graph:
    """Generate a distance-biased graph suitable for topology experiments.

    This is a lightweight Waxman-style generator. It is not a replacement for
    real Internet topologies; it creates synthetic baselines with geography-like
    locality for controlled experiments.
    """

    if alpha <= 0.0:
        raise ValueError("Alpha must be positive")

    rng = Random(seed)
    positions = [(rng.random(), rng.random()) for _ in range(node_count)]
    max_distance = 2**0.5
    edges: list[tuple[int, int]] = []

    for i in range(node_count):
        for j in range(i + 1, node_count):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            distance = (dx * dx + dy * dy) ** 0.5
            probability = min(1.0, alpha * (1.0 - distance / max_distance))
            if rng.random() <= probability:
                edges.append((i, j))

    graph = Graph.from_tuples(node_count, edges, reliability)
    if ensure_connected and not graph.is_connected():
        edges.extend(_connection_edges(graph, rng))
    return Graph.from_tuples(node_count, edges, reliability)


def barabasi_albert_graph(
    node_count: int,
    attach_edges: int,
    reliability: float,
    seed: int | None = None,
) -> Graph:
    """Generate a preferential-attachment graph as a synthetic AS-like baseline."""

    if node_count <= 0:
        raise ValueError("Node count must be positive")
    if attach_edges <= 0:
        raise ValueError("Attach edge count must be positive")
    if attach_edges >= node_count:
        raise ValueError("Attach edge count must be smaller than node count")

    rng = Random(seed)
    edges: list[tuple[int, int]] = []
    degrees = [0 for _ in range(node_count)]

    for node in range(1, attach_edges + 1):
        edges.append((0, node))
        degrees[0] += 1
        degrees[node] += 1

    for node in range(attach_edges + 1, node_count):
        targets: set[int] = set()
        weighted_nodes = [idx for idx, degree in enumerate(degrees[:node]) for _ in range(max(1, degree))]
        while len(targets) < attach_edges:
            targets.add(rng.choice(weighted_nodes))
        for target in sorted(targets):
            edges.append((target, node))
            degrees[target] += 1
            degrees[node] += 1

    return Graph.from_tuples(node_count, edges, reliability)


def _connection_edges(graph: Graph, rng: Random) -> list[tuple[int, int]]:
    adjacency = graph.neighbors()
    remaining = set(range(graph.node_count))
    components: list[list[int]] = []

    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.append(neighbor)
                    stack.append(neighbor)
        components.append(component)

    added: list[tuple[int, int]] = []
    for left, right in zip(components, components[1:]):
        added.append((rng.choice(left), rng.choice(right)))
    return added
