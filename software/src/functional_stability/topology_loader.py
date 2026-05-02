"""Load public communication/network topology datasets from local files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from .graph import Graph


@dataclass(frozen=True)
class LoadedTopology:
    graph: Graph
    source: str
    node_labels: tuple[str, ...]


def load_graphml(path: str | Path, reliability: float = 0.99, source: str | None = None) -> LoadedTopology:
    """Load GraphML files such as Internet Topology Zoo exports."""

    path = Path(path)
    root = ElementTree.parse(path).getroot()
    namespace = ""
    if root.tag.startswith("{"):
        namespace = root.tag.split("}", 1)[0] + "}"

    graph_element = root.find(f".//{namespace}graph")
    if graph_element is None:
        raise ValueError("GraphML file does not contain a graph element")

    nodes = [node.attrib["id"] for node in graph_element.findall(f"{namespace}node")]
    node_map = {node_id: index for index, node_id in enumerate(nodes)}
    edges: list[tuple[int, int]] = []

    for edge in graph_element.findall(f"{namespace}edge"):
        source_id = edge.attrib.get("source")
        target_id = edge.attrib.get("target")
        if source_id in node_map and target_id in node_map and source_id != target_id:
            edges.append((node_map[source_id], node_map[target_id]))

    return LoadedTopology(
        graph=Graph.from_tuples(len(nodes), _unique_edges(edges), reliability),
        source=source or f"graphml:{path.name}",
        node_labels=tuple(nodes),
    )


def load_topology_zoo_gml(
    path: str | Path,
    reliability: float = 0.99,
    source: str | None = None,
) -> LoadedTopology:
    """Load simple GML files from Internet Topology Zoo or TopoHub exports."""

    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    current_block: str | None = None
    fields: dict[str, str] = {}

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("node"):
            current_block = "node"
            fields = {}
            continue
        if line.startswith("edge"):
            current_block = "edge"
            fields = {}
            continue
        if line == "]" and current_block:
            if current_block == "node" and "id" in fields:
                nodes[fields["id"]] = fields.get("label", fields["id"])
            if current_block == "edge" and "source" in fields and "target" in fields:
                edges.append((fields["source"], fields["target"]))
            current_block = None
            fields = {}
            continue
        if current_block:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                fields[parts[0]] = parts[1].strip('"')

    node_ids = list(nodes.keys())
    node_map = {node_id: index for index, node_id in enumerate(node_ids)}
    edge_tuples = [
        (node_map[source_id], node_map[target_id])
        for source_id, target_id in edges
        if source_id in node_map and target_id in node_map and source_id != target_id
    ]

    return LoadedTopology(
        graph=Graph.from_tuples(len(node_ids), _unique_edges(edge_tuples), reliability),
        source=source or f"gml:{path.name}",
        node_labels=tuple(nodes[node_id] for node_id in node_ids),
    )


def load_caida_as_relationships(
    path: str | Path,
    reliability: float = 0.99,
    source: str | None = None,
) -> LoadedTopology:
    """Load CAIDA AS Relationships files.

    Expected line formats include:
    provider-as|customer-as|-1
    peer-as|peer-as|0|source
    Comment lines starting with '#' are ignored.
    """

    path = Path(path)
    as_numbers: dict[str, int] = {}
    edges: list[tuple[int, int]] = []

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        left, right = parts[0], parts[1]
        if left == right:
            continue
        for asn in (left, right):
            if asn not in as_numbers:
                as_numbers[asn] = len(as_numbers)
        edges.append((as_numbers[left], as_numbers[right]))

    labels = [None for _ in as_numbers]
    for asn, index in as_numbers.items():
        labels[index] = asn

    return LoadedTopology(
        graph=Graph.from_tuples(len(as_numbers), _unique_edges(edges), reliability),
        source=source or f"caida-as:{path.name}",
        node_labels=tuple(label or "" for label in labels),
    )


def load_topology(path: str | Path, topology_type: str, reliability: float = 0.99) -> LoadedTopology:
    if topology_type == "graphml":
        return load_graphml(path, reliability=reliability)
    if topology_type == "gml":
        return load_topology_zoo_gml(path, reliability=reliability)
    if topology_type == "caida-as":
        return load_caida_as_relationships(path, reliability=reliability)
    raise ValueError(f"Unsupported topology type: {topology_type}")


def _unique_edges(edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    unique: set[tuple[int, int]] = set()
    for source, target in edges:
        if source == target:
            continue
        unique.add(tuple(sorted((source, target))))
    return sorted(unique)
