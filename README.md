# Network Reliability Analytics

Research-oriented Python software for analysing communication-network
reliability, generating graph datasets, and evaluating machine-learning
approximations. The project keeps the research workflow as its core and
provides a small FastAPI service for submitting topology analyses and
retrieving persisted results.

## What it demonstrates

- Exact all-terminal reliability calculations for small undirected graphs.
- Synthetic and real topology processing, deterministic data splits, and
  reproducible baseline/CNN experiments.
- A typed Python package with automated tests and command-line workflows.
- A lightweight REST API with request validation, structured logs, consistent
  error responses, environment-based configuration, PostgreSQL persistence,
  and Redis/RQ background processing.

## Architecture

```text
FastAPI API  ──► PostgreSQL (analysis requests and results)
     │
     └────────► Redis queue ──► RQ worker ──► exact reliability engine
```

The API is deliberately small: it is an interface around the analysis package,
not a replacement for the research/data-analysis project.

## Quick start

### Full local stack (recommended)

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and run:

```bash
docker compose up --build
```

This starts the API at `http://localhost:8000`, interactive OpenAPI docs at
`http://localhost:8000/docs`, PostgreSQL, Redis, and an RQ worker. PostgreSQL
data is retained in the `postgres_data` Docker volume.

Submit a small triangle topology:

```bash
curl -X POST http://localhost:8000/api/v1/analyses \
  -H "Content-Type: application/json" \
  -d '{"sample_id":"triangle-001","node_count":3,"edges":[{"source":0,"target":1,"reliability":0.99},{"source":1,"target":2,"reliability":0.99},{"source":0,"target":2,"reliability":0.99}]}'
```

### Analysis package and experiments

```bash
cd software
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -q
```

See [software/README.md](software/README.md) for data-generation, baseline,
and CNN evaluation commands. Install `.[ml]` when running the PyTorch
experiments.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Database-aware health check |
| `POST` | `/api/v1/analyses` | Queue an exact network-reliability analysis |
| `GET` | `/api/v1/analyses` | List recent analyses |
| `GET` | `/api/v1/analyses/{id}` | Retrieve one analysis and its status/result |

Exact calculations enumerate edge states and are therefore limited to 20 edges
by default. Configure `NR_MAX_EXACT_EDGES` only with care.

## Configuration

Copy `.env.example` to `.env` for non-Docker development. Settings use the
`NR_` prefix, including `NR_DATABASE_URL`, `NR_REDIS_URL`,
`NR_USE_INLINE_JOBS`, `NR_LOG_LEVEL`, and `NR_MAX_EXACT_EDGES`.

For database schema changes, use Alembic from the repository root:

```bash
alembic upgrade head
```

## Development quality checks

```bash
cd software
ruff check src tests
pytest -q
```

GitHub Actions runs the same checks and verifies that the Docker services build.

## Limitations

- Exact all-terminal reliability is exponential in the number of edges.
- The ML experiments are deliberately small and their results should not be
  treated as general evidence that a CNN outperforms traditional models.
- The API currently targets demonstration-scale topology-analysis jobs rather
  than multi-tenant or high-throughput production use.
