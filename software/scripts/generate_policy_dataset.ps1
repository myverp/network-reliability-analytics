param(
    [string]$PythonPath = "src"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = $PythonPath

New-Item -ItemType Directory -Force -Path "outputs/policy" | Out-Null

# Synthetic baseline samples. Increase the loop count after validating runtime.
$reliabilities = @(0.95, 0.97, 0.99)
$sizes = @(6, 8, 10)
$models = @("erdos-renyi", "waxman", "barabasi-albert")

foreach ($reliability in $reliabilities) {
    foreach ($size in $sizes) {
        foreach ($model in $models) {
            foreach ($index in 0..9) {
                $sampleId = "syn-$model-n$size-r$($reliability.ToString('0.00'))-$index"
                $jsonl = "outputs/policy/$sampleId.jsonl"
                $metadata = "outputs/policy/$sampleId.csv"

                $commonArgs = @(
                    "-m", "functional_stability.cli",
                    "--sample-id", $sampleId,
                    "--source", "synthetic",
                    "--synthetic-model", $model,
                    "--nodes", "$size",
                    "--edge-reliability", "$reliability",
                    "--seed", "$((10000 * ($models.IndexOf($model) + 1)) + $index)",
                    "--matrix-size", "10",
                    "--max-label-edges", "20",
                    "--output-jsonl", $jsonl,
                    "--output-metadata", $metadata
                )

                if ($model -eq "erdos-renyi") {
                    python @commonArgs --edge-probability 0.3
                } elseif ($model -eq "waxman") {
                    python @commonArgs --waxman-alpha 0.35
                } else {
                    python @commonArgs --attach-edges 2
                }
            }
        }
    }
}

# Real topology samples from Internet Topology Zoo Abilene.
foreach ($reliability in $reliabilities) {
    foreach ($size in $sizes) {
        foreach ($index in 0..1) {
            $sampleId = "real-abilene-bfs-n$size-r$($reliability.ToString('0.00'))-$index"
            python -m functional_stability.cli `
                --sample-id $sampleId `
                --source file `
                --topology-file data/topology_zoo/Abilene.graphml `
                --topology-type graphml `
                --edge-reliability $reliability `
                --subgraph-method bfs `
                --start-node $index `
                --subgraph-nodes $size `
                --matrix-size 10 `
                --max-label-edges 20 `
                --output-jsonl "outputs/policy/$sampleId.jsonl" `
                --output-metadata "outputs/policy/$sampleId.csv"

            $sampleId = "real-abilene-random-n$size-r$($reliability.ToString('0.00'))-$index"
            python -m functional_stability.cli `
                --sample-id $sampleId `
                --source file `
                --topology-file data/topology_zoo/Abilene.graphml `
                --topology-type graphml `
                --edge-reliability $reliability `
                --subgraph-method random `
                --subgraph-nodes $size `
                --seed "$((40000 + $index))" `
                --matrix-size 10 `
                --max-label-edges 20 `
                --output-jsonl "outputs/policy/$sampleId.jsonl" `
                --output-metadata "outputs/policy/$sampleId.csv"
        }
    }
}
