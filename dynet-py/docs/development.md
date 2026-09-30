# Development

## Local Setup

```bash
git clone <your-repo-url>
cd dynet-py
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,docs]"
```

## Run Tests

```bash
python -m pytest -q
```

## Compare Rewiring Backends

```bash
MPLBACKEND=Agg PYTHONPATH=src python benchmarks/rewiring_backends.py \
  --nodes 500 --networks 8 --densities 0.001 0.01 0.1 0.5 --repeats 5
```

The seeded benchmark checks dense/sparse numerical agreement and prints one JSON
record per backend and density. Input generation and preparation are excluded.
First-call timings include cache construction; repeated timings report the
median. Peak traced allocations are measured separately on fresh prepared inputs
and include calculation caches and temporaries, not total process RSS.

The automatic size/density threshold is a heuristic. Sparse storage can take
more memory and time at high density; compare representative data before forcing
a backend. Very large dense benchmarks can require substantial memory.
