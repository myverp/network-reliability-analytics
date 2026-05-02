"""Command-line interface for dataset preparation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import export_jsonl, export_metadata_csv, make_sample
from .graph import Graph
from .subgraph import bfs_subgraph, largest_connected_component, random_connected_subgraph
from .synthetic import barabasi_albert_graph, erdos_renyi_graph, waxman_like_graph
from .topology_loader import load_topology


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    graph, source = _load_graph(args)
    graph = _apply_subgraph(args, graph)

    matrix_size = args.matrix_size or graph.node_count
    sample = make_sample(
        sample_id=args.sample_id,
        source=source,
        graph=graph,
        matrix_size=matrix_size,
        max_label_edges=args.max_label_edges,
    )

    export_jsonl([sample], args.output_jsonl)
    export_metadata_csv([sample], args.output_metadata)

    label = "skipped" if sample.label is None else f"{sample.label:.12g}"
    print(
        "prepared sample "
        f"id={sample.sample_id} source={sample.source} "
        f"nodes={sample.graph.node_count} edges={len(sample.graph.edges)} "
        f"matrix={matrix_size} label={label} method={sample.label_method}"
    )
    print(f"jsonl={Path(args.output_jsonl)}")
    print(f"metadata={Path(args.output_metadata)}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare adjacency-matrix datasets for all-terminal reliability experiments."
    )
    parser.add_argument("--sample-id", required=True, help="Stable sample identifier written to exports.")
    parser.add_argument(
        "--source",
        choices=["synthetic", "file"],
        required=True,
        help="Graph source type.",
    )
    parser.add_argument("--edge-reliability", type=float, default=0.99, help="Default edge reliability.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible generation.")

    parser.add_argument(
        "--synthetic-model",
        choices=["erdos-renyi", "waxman", "barabasi-albert"],
        default="erdos-renyi",
        help="Synthetic graph model.",
    )
    parser.add_argument("--nodes", type=int, default=10, help="Synthetic node count.")
    parser.add_argument("--edge-probability", type=float, default=0.3, help="Erdos-Renyi edge probability.")
    parser.add_argument("--attach-edges", type=int, default=2, help="Barabasi-Albert edges per new node.")
    parser.add_argument("--waxman-alpha", type=float, default=0.35, help="Waxman-like locality parameter.")

    parser.add_argument("--topology-file", help="Local real topology file path.")
    parser.add_argument(
        "--topology-type",
        choices=["graphml", "gml", "caida-as"],
        help="Input file format for --source file.",
    )

    parser.add_argument(
        "--subgraph-method",
        choices=["none", "largest", "bfs", "random"],
        default="none",
        help="Optional subgraph extraction method.",
    )
    parser.add_argument("--subgraph-nodes", type=int, help="Maximum nodes for bfs/random subgraphs.")
    parser.add_argument("--start-node", type=int, default=0, help="Start node for BFS subgraph extraction.")

    parser.add_argument("--matrix-size", type=int, help="Fixed adjacency matrix size. Defaults to graph size.")
    parser.add_argument(
        "--max-label-edges",
        type=int,
        default=20,
        help="Maximum edge count for exact all-terminal reliability labeling.",
    )
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL dataset path.")
    parser.add_argument("--output-metadata", required=True, help="Output CSV metadata path.")
    return parser


def _load_graph(args: argparse.Namespace) -> tuple[Graph, str]:
    if args.source == "synthetic":
        return _load_synthetic_graph(args)

    if not args.topology_file or not args.topology_type:
        raise SystemExit("--topology-file and --topology-type are required for --source file")
    loaded = load_topology(args.topology_file, args.topology_type, reliability=args.edge_reliability)
    return loaded.graph, loaded.source


def _load_synthetic_graph(args: argparse.Namespace) -> tuple[Graph, str]:
    if args.synthetic_model == "erdos-renyi":
        graph = erdos_renyi_graph(
            node_count=args.nodes,
            edge_probability=args.edge_probability,
            reliability=args.edge_reliability,
            seed=args.seed,
            ensure_connected=True,
        )
    elif args.synthetic_model == "waxman":
        graph = waxman_like_graph(
            node_count=args.nodes,
            reliability=args.edge_reliability,
            alpha=args.waxman_alpha,
            seed=args.seed,
        )
    elif args.synthetic_model == "barabasi-albert":
        graph = barabasi_albert_graph(
            node_count=args.nodes,
            attach_edges=args.attach_edges,
            reliability=args.edge_reliability,
            seed=args.seed,
        )
    else:
        raise SystemExit(f"Unsupported synthetic model: {args.synthetic_model}")
    return graph, f"synthetic:{args.synthetic_model}"


def _apply_subgraph(args: argparse.Namespace, graph: Graph) -> Graph:
    if args.subgraph_method == "none":
        return graph
    if args.subgraph_method == "largest":
        return largest_connected_component(graph)
    if args.subgraph_method == "bfs":
        if args.subgraph_nodes is None:
            raise SystemExit("--subgraph-nodes is required for BFS subgraph extraction")
        return bfs_subgraph(graph, start_node=args.start_node, max_nodes=args.subgraph_nodes)
    if args.subgraph_method == "random":
        if args.subgraph_nodes is None:
            raise SystemExit("--subgraph-nodes is required for random subgraph extraction")
        return random_connected_subgraph(graph, max_nodes=args.subgraph_nodes, seed=args.seed)
    raise SystemExit(f"Unsupported subgraph method: {args.subgraph_method}")


if __name__ == "__main__":
    raise SystemExit(main())
