# Software Part

This directory contains the implementation for the thesis software component.

The current scope is based on the written thesis materials:

- model a network information system as an undirected graph;
- represent the graph as an adjacency matrix/tensor input for later CNN work;
- compute exact all-terminal network reliability for small graphs as reference values;
- use those reference values later for dataset generation, CNN training, and comparison.

The first implementation stage intentionally does not include CNN training yet. A trained
model needs a reliable source of target values first.

## Current Modules

- `functional_stability.graph`: graph data structures and simple graph builders.
- `functional_stability.reliability`: exact all-terminal reliability calculation.
- `functional_stability.synthetic`: synthetic topology generation.
- `functional_stability.topology_loader`: local loaders for public topology files.
- `functional_stability.subgraph`: connected subgraph extraction for large topologies.
- `functional_stability.dataset`: adjacency preparation, feasible exact labeling, and export.

## Real Topology Sources

Use datasets that describe communication, Internet, backbone, AS-level, or laboratory
network topology. Do not use social-network datasets for the thesis experiments.

Recommended sources:

- Internet Topology Zoo (`https://topology-zoo.org/dataset.html`): provider, research/education, and backbone network
  topologies. The dataset page provides GML and GraphML files, which are directly
  relevant to communication network topology work.
- CAIDA AS Relationships (`https://www.caida.org/catalog/datasets/as-relationships/`):
  AS-level Internet topology inferred from BGP, RouteViews,
  RIPE RIS, and related measurements. Use it for large AS-level graph experiments.
  The original data is large, so exact all-terminal reliability should be computed
  only on small extracted subgraphs.
- TopoHub (`https://www.topohub.org/`): a networking-research topology repository that aggregates Topology Zoo,
  CAIDA, SNDlib, backbone, and synthetic Gabriel graph topologies. It is useful as
  a practical access layer and for additional laboratory/reference topologies.

Supported local file loaders:

- `load_graphml(path)`: GraphML, suitable for Topology Zoo GraphML exports.
- `load_topology_zoo_gml(path)`: simple GML, suitable for Topology Zoo or TopoHub
  GML exports.
- `load_caida_as_relationships(path)`: CAIDA pipe-delimited AS relationship files.

Keep downloaded datasets under `software/data/`. That directory is ignored by git.

## Dataset Preparation

The current pipeline supports:

- synthetic graph generation with Erdos-Renyi, Waxman-like, and Barabasi-Albert
  generators;
- loading real topology files from local GraphML, GML, or CAIDA AS relationship data;
- extracting connected subgraphs from large real graphs;
- creating fixed-size padded adjacency matrices;
- exact all-terminal reliability labeling only when the edge count is feasible;
- JSONL export with adjacency matrices and labels;
- CSV metadata export.

Exact labeling uses exhaustive edge-state enumeration. Keep `max_label_edges` small
for real experiments. Larger real graphs should be sliced into subgraphs first.

Programmatic example:

```python
from functional_stability.dataset import export_jsonl, export_metadata_csv, make_sample
from functional_stability.subgraph import random_connected_subgraph
from functional_stability.topology_loader import load_graphml

loaded = load_graphml("software/data/Abilene.graphml", reliability=0.99)
subgraph = random_connected_subgraph(loaded.graph, max_nodes=10, seed=1)
sample = make_sample("abilene-001", loaded.source, subgraph, matrix_size=10, max_label_edges=20)

export_jsonl([sample], "software/outputs/dataset.jsonl")
export_metadata_csv([sample], "software/outputs/metadata.csv")
```

## CLI Usage

Run the CLI from the `software` directory with `PYTHONPATH` pointed at `src`.

PowerShell:

```powershell
cd software
$env:PYTHONPATH = "src"
```

Synthetic graph example:

```powershell
python -m functional_stability.cli `
  --sample-id synthetic-er-001 `
  --source synthetic `
  --synthetic-model erdos-renyi `
  --nodes 8 `
  --edge-probability 0.3 `
  --edge-reliability 0.99 `
  --seed 1 `
  --matrix-size 8 `
  --max-label-edges 20 `
  --output-jsonl outputs/synthetic_er_dataset.jsonl `
  --output-metadata outputs/synthetic_er_metadata.csv
```

Real Topology Zoo GraphML example:

```powershell
python -m functional_stability.cli `
  --sample-id abilene-topozoo-bfs-8 `
  --source file `
  --topology-file data/topology_zoo/Abilene.graphml `
  --topology-type graphml `
  --edge-reliability 0.99 `
  --subgraph-method bfs `
  --start-node 0 `
  --subgraph-nodes 8 `
  --matrix-size 8 `
  --max-label-edges 20 `
  --output-jsonl outputs/abilene_topozoo_sample.jsonl `
  --output-metadata outputs/abilene_topozoo_metadata.csv
```

Supported `--subgraph-method` values:

- `none`: use the whole graph;
- `largest`: use the largest connected component;
- `bfs`: extract a connected BFS subgraph from `--start-node`;
- `random`: extract a connected subgraph from a seeded random start node.

The Abilene sample prepared for the first real experiment uses Internet Topology Zoo:

- source file: `software/data/topology_zoo/Abilene.graphml`;
- output JSONL: `software/outputs/abilene_topozoo_sample.jsonl`;
- output metadata: `software/outputs/abilene_topozoo_metadata.csv`;
- sample shape: 8 nodes, 9 edges, 8x8 adjacency matrix;
- label: exact all-terminal reliability with edge reliability `0.99`.

The `software/data/` and `software/outputs/` directories are ignored by git because
source datasets and generated experiment outputs should be reproducible artifacts, not
repository source code.

## Run Tests

```powershell
cd software
python -m unittest discover -s tests
```

## Planned Stack

The thesis materials justify Python with NetworkX, NumPy, PyTorch, scikit-learn, and
matplotlib. The current dataset foundation still uses only the Python standard library
so it can be tested before the full CNN stack is installed.
