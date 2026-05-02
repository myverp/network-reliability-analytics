"""Exact all-terminal network reliability for small undirected graphs."""

from __future__ import annotations

from collections import deque

from .graph import Edge, Graph


def all_terminal_reliability(graph: Graph) -> float:
    """Return probability that all graph nodes remain connected.

    This exact implementation enumerates all edge states, so it is suitable only
    for small graphs. It provides reference target values for later datasets.
    """

    if graph.node_count == 1:
        return 1.0

    edge_count = len(graph.edges)
    if edge_count == 0:
        return 0.0

    total = 0.0
    for mask in range(1 << edge_count):
        active_edges: list[Edge] = []
        state_probability = 1.0

        for index, edge in enumerate(graph.edges):
            edge_is_active = bool(mask & (1 << index))
            if edge_is_active:
                active_edges.append(edge)
                state_probability *= edge.reliability
            else:
                state_probability *= 1.0 - edge.reliability

        if state_probability and _is_connected(graph.node_count, active_edges):
            total += state_probability

    return total


def _is_connected(node_count: int, edges: list[Edge]) -> bool:
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for edge in edges:
        adjacency[edge.source].append(edge.target)
        adjacency[edge.target].append(edge.source)

    visited = {0}
    queue: deque[int] = deque([0])

    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == node_count
