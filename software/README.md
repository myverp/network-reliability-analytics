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
- `functional_stability.cnn_smoke`: minimal PyTorch CNN smoke experiment for
  reliability regression.

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

## Experimental Dataset Policy

The first reproducible pre-CNN dataset policy is stored in `dataset_policy.json`.
It is intentionally small enough for exact all-terminal reliability labeling while still
covering several topology patterns.

Chosen setup:

- matrix size: `10x10`;
- graph sizes: `6`, `8`, and `10` nodes;
- edge reliability values: `0.95`, `0.97`, and `0.99`;
- exact label limit: at most `20` edges, with disconnected graphs labeled as `0.0`;
- synthetic models:
  - Erdos-Renyi with `p = 0.3`, connected by construction;
  - Waxman-like with `alpha = 0.35`, connected by construction;
  - Barabasi-Albert with `m = 2`;
- synthetic dataset size: `270` samples:
  `3 models * 3 graph sizes * 3 reliability values * 10 seeds`;
- real topology source for the first policy run: Internet Topology Zoo Abilene GraphML;
- real subgraph sizes: `6`, `8`, and `10` nodes;
- real subgraph methods: deterministic BFS and seeded random connected subgraph;
- real dataset size for one source: `36` samples:
  `3 graph sizes * 3 reliability values * 2 methods * 2 samples`;
- split policy: deterministic stratified split by graph family/source, then stable
  hash ordering by `sample_id` inside each group; use `70%` train, `15%` validation,
  `15%` test.

The split is performed by the assembly CLI. It is stratified to avoid a biased split
where one graph family appears only in validation or test.

Repeatable generation command:

```powershell
cd software
.\scripts\generate_policy_dataset.ps1
```

The script writes one JSONL and one metadata CSV per generated sample under
`software/outputs/policy/`. Combining these per-sample files into train/validation/test
files is the next dataset-engineering step.

Quick policy smoke checks:

```powershell
$env:PYTHONPATH = "src"
python -m functional_stability.cli --sample-id syn-erdos-renyi-n8-r0.99-0 --source synthetic --synthetic-model erdos-renyi --nodes 8 --edge-probability 0.3 --edge-reliability 0.99 --seed 10000 --matrix-size 10 --max-label-edges 20 --output-jsonl outputs/policy_smoke_syn.jsonl --output-metadata outputs/policy_smoke_syn.csv
python -m functional_stability.cli --sample-id real-abilene-bfs-n8-r0.99-0 --source file --topology-file data/topology_zoo/Abilene.graphml --topology-type graphml --edge-reliability 0.99 --subgraph-method bfs --start-node 0 --subgraph-nodes 8 --matrix-size 10 --max-label-edges 20 --output-jsonl outputs/policy_smoke_real.jsonl --output-metadata outputs/policy_smoke_real.csv
```

## Dataset Assembly And Split

After sample generation, assemble validated samples into deterministic split files:

```powershell
$env:PYTHONPATH = "src"
python -m functional_stability.assemble_cli `
  --input-glob "outputs/policy/*.jsonl" `
  --policy dataset_policy.json `
  --output-dir outputs/final_policy_dataset
```

For a quick smoke split using the two policy smoke samples:

```powershell
python -m functional_stability.assemble_cli `
  --input-glob "outputs/policy_smoke_*.jsonl" `
  --policy dataset_policy.json `
  --output-dir outputs/final_policy_smoke
```

The assembly stage validates:

- duplicate `sample_id` values;
- missing labels;
- skipped/non-exact labels unless explicitly allowed;
- matrix size consistency with `dataset_policy.json`;
- valid node and edge counts.

The output directory contains:

- `train.jsonl`, `validation.jsonl`, `test.jsonl`;
- `train_metadata.csv`, `validation_metadata.csv`, `test_metadata.csv`;
- `assembly_report.json`.

The split is deterministic and stratified: samples are grouped by graph family/source,
ordered by a stable hash of `sample_id` inside each group, and then divided according
to the policy ratios. The hash ordering avoids accidental train/validation/test bias
from lexicographic sample names such as `n10`, `n6`, and `n8`.

## Baseline Evaluation

Before CNN training, run simple non-CNN baselines on the final split files:

```powershell
$env:PYTHONPATH = "src"
python -m functional_stability.baseline_cli `
  --train outputs/final_policy_dataset/train.jsonl `
  --validation outputs/final_policy_dataset/validation.jsonl `
  --test outputs/final_policy_dataset/test.jsonl `
  --output outputs/final_policy_dataset/baseline_report.json `
  --ridge-alpha 1.0
```

The baseline stage evaluates:

- mean-label predictor;
- ridge regression over flattened adjacency matrices.

Current local baseline result on the full policy dataset:

- mean baseline test MAE: `0.052848`;
- mean baseline test RMSE: `0.067589`;
- ridge baseline test MAE: `0.035949`;
- ridge baseline test RMSE: `0.048865`.

Any CNN model should beat the ridge baseline, not only the mean baseline.

## Minimal CNN Smoke Experiment

The first CNN stage is intentionally small. It checks that the final `10x10`
adjacency matrices can be consumed by a PyTorch regression model and compared with
the current ridge baseline. This is not thesis-scale tuning.

Install dependencies:

```powershell
cd software
python -m pip install -r requirements.txt
```

If only the missing CNN dependency is needed:

```powershell
python -m pip install torch
```

Train and evaluate the smoke model comparison:

```powershell
$env:PYTHONPATH = "src"
python -m functional_stability.cnn_smoke_cli `
  --train outputs/final_policy_dataset/train.jsonl `
  --validation outputs/final_policy_dataset/validation.jsonl `
  --test outputs/final_policy_dataset/test.jsonl `
  --output outputs/final_policy_dataset/cnn_smoke_report.json `
  --epochs 50 `
  --batch-size 32 `
  --learning-rate 0.001 `
  --seed 42 `
  --early-stopping-patience 20 `
  --ridge-test-mae 0.035949
```

The model input shape is `1 x 10 x 10`; the output is one predicted reliability
value. The report is written to `outputs/final_policy_dataset/cnn_smoke_report.json`
and includes validation/test MAE, validation/test RMSE, training settings, model
parameter counts, and whether the best neural model beats ridge test MAE `0.035949`.
Inputs are standardized using the train split mean and standard deviation. Training
uses early stopping on validation MAE.

Current smoke candidates:

- `small_cnn`: `Conv2d(1,8,3,pad=1) -> ReLU -> Conv2d(8,16,3,pad=1) -> ReLU -> AdaptiveAvgPool2d(1,1) -> Linear(16,16) -> ReLU -> Linear(16,1)`;
- `stronger_cnn`: `Conv2d(1,16,3,pad=1) -> ReLU -> Conv2d(16,32,3,pad=1) -> ReLU -> AdaptiveAvgPool2d(2,2) -> Linear(128,32) -> ReLU -> Linear(32,1)`;
- `mlp`: `Flatten(10x10) -> Linear(100,64) -> ReLU -> Linear(64,32) -> ReLU -> Linear(32,1)`.

Best current local result with `--epochs 300`, `--early-stopping-patience 30`,
and seed `42`:

- `small_cnn`: validation MAE `0.039054`, validation RMSE `0.049886`,
  test MAE `0.040580`, test RMSE `0.053457`;
- `stronger_cnn`: validation MAE `0.028870`, validation RMSE `0.044272`,
  test MAE `0.031942`, test RMSE `0.048117`;
- `mlp`: validation MAE `0.042460`, validation RMSE `0.090848`,
  test MAE `0.051892`, test RMSE `0.124082`.

The current best model is `stronger_cnn`, which beats ridge test MAE `0.035949`
on this smoke split.

### Stronger CNN Multi-Seed Evaluation

After selecting `stronger_cnn`, run the same dataset and hyperparameters across
several fixed random seeds:

```powershell
$env:PYTHONPATH = "src"
python -m functional_stability.cnn_multiseed_cli `
  --train outputs/final_policy_dataset/train.jsonl `
  --validation outputs/final_policy_dataset/validation.jsonl `
  --test outputs/final_policy_dataset/test.jsonl `
  --output outputs/final_policy_dataset/stronger_cnn_multiseed_report.json `
  --seeds 42,43,44,45,46 `
  --epochs 300 `
  --batch-size 32 `
  --learning-rate 0.001 `
  --early-stopping-patience 30 `
  --ridge-test-mae 0.035949
```

The report format contains:

- `per_seed`: validation/test MAE and RMSE, best epoch, training time, and whether
  the seed beats ridge test MAE;
- `summary`: mean and standard deviation for validation/test MAE and RMSE;
- `consistently_beats_ridge_test_mae`: true only when every seed beats ridge test
  MAE `0.035949`.

Current local five-seed result for seeds `42,43,44,45,46`:

- validation MAE mean/std: `0.029452` / `0.002063`;
- validation RMSE mean/std: `0.043867` / `0.003215`;
- test MAE mean/std: `0.035027` / `0.002533`;
- test RMSE mean/std: `0.050275` / `0.001152`;
- consistently beats ridge test MAE `0.035949`: `false`.

### Repeated Stronger CNN vs Ridge Evaluation

For thesis-quality reporting, compare the selected neural model with ridge on the
same train/validation/test split using repeated fixed neural seeds:

```powershell
$env:PYTHONPATH = "src"
python -m functional_stability.repeated_eval_cli `
  --train outputs/final_policy_dataset/train.jsonl `
  --validation outputs/final_policy_dataset/validation.jsonl `
  --test outputs/final_policy_dataset/test.jsonl `
  --output outputs/final_policy_dataset/repeated_stronger_cnn_vs_ridge_report.json `
  --seeds 42,43,44,45,46,47,48,49,50,51 `
  --epochs 300 `
  --batch-size 32 `
  --learning-rate 0.001 `
  --early-stopping-patience 30 `
  --ridge-alpha 1.0 `
  --ridge-test-mae 0.035949
```

The report includes:

- per-seed `stronger_cnn` validation/test MAE and RMSE;
- ridge validation/test MAE and RMSE;
- mean, sample standard deviation, and approximate 95% confidence intervals;
- separate synthetic and real topology metrics when `sample_id`/`source` metadata
  identifies the sample family.

Current 10-seed result for seeds `42..51`:

- `stronger_cnn` test MAE mean/std/95% CI:
  `0.034135` / `0.002674` / `[0.032477, 0.035792]`;
- `stronger_cnn` test RMSE mean/std/95% CI:
  `0.049610` / `0.001460` / `[0.048705, 0.050514]`;
- ridge test MAE: `0.035949`;
- ridge test RMSE: `0.048865`;
- mean test MAE difference, CNN minus ridge: `-0.001814`.

By topology family on the test split:

- real samples: `stronger_cnn` MAE `0.036803` vs ridge MAE `0.044578`;
- synthetic samples: `stronger_cnn` MAE `0.033660` vs ridge MAE `0.034414`.

Conclusion for the current small dataset: `stronger_cnn` is slightly better than ridge
on mean test MAE, especially on real Topology Zoo samples, but the margin is small
and the neural confidence interval overlaps the ridge value. Treat this as promising,
not yet definitive.

If PyTorch is not installed, the CLI exits with a clear install message and no
training report is produced.

## Run Tests

```powershell
cd software
python -m unittest discover -s tests
```

## Planned Stack

The thesis materials justify Python with NetworkX, NumPy, PyTorch, scikit-learn, and
matplotlib. The dataset and baseline stages use NumPy. The CNN smoke stage requires
PyTorch.
