"""Subgraph extraction utilities for large real topologies."""

from __future__ import annotations

from collections import deque
from random import Random

from .graph import Graph


def largest_connected_component(graph: Graph) -> Graph:
    components = _connected_components(graph)
    largest = max(components, key=len)
    return graph.relabeled_subgraph(sorted(largest))


def bfs_subgraph(graph: Graph, start_node: int, max_nodes: int) -> Graph:
    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive")
    if start_node < 0 or start_node >= graph.node_count:
        raise ValueError("start_node is outside graph node range")

    adjacency = graph.neighbors()
    selected: list[int] = []
    seen = {start_node}
    queue: deque[int] = deque([start_node])

    while queue and len(selected) < max_nodes:
        node = queue.popleft()
        selected.append(node)
        for neighbor in sorted(adjacency[node]):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)

    return graph.relabeled_subgraph(selected)


def random_connected_subgraph(graph: Graph, max_nodes: int, seed: int | None = None) -> Graph:
    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive")

    base = largest_connected_component(graph)
    rng = Random(seed)
    start = rng.randrange(base.node_count)
    return bfs_subgraph(base, start, min(max_nodes, base.node_count))


def _connected_components(graph: Graph) -> list[set[int]]:
    adjacency = graph.neighbors()
    remaining = set(range(graph.node_count))
    components: list[set[int]] = []

    while remaining:
        start = remaining.pop()
        component = {start}
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)

    return components
