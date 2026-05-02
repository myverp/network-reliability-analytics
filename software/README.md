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

Minimal example:

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

## Run Tests

```powershell
cd software
python -m unittest discover -s tests
```

## Planned Stack

The thesis materials justify Python with NetworkX, NumPy, PyTorch, scikit-learn, and
matplotlib. The current dataset foundation still uses only the Python standard library
so it can be tested before the full CNN stack is installed.
