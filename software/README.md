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

## Run Tests

```powershell
cd software
python -m unittest discover -s tests
```

## Planned Stack

The thesis materials justify Python with NetworkX, NumPy, PyTorch, scikit-learn, and
matplotlib. The initial foundation uses only the Python standard library so it can be
tested before the full research stack is installed.
