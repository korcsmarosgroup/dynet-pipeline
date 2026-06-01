# dynet-py

`dynet-py` is a Python package for network rewiring analysis.

It installs the Python import package `dynet_py` and the CLI command `dynet-py`.

It exposes the main API:

- `format_indata`
- `rewiring_analysis`
- `rewiring_plot`
- `small_multiples_plot`
- `calculate_jaccard_indices`
- `compare_targeting`

It also keeps the earlier draft API:

- `package_data`
- `package_data_rename`
- `package_data_remap`
- `dynet_internal`
- `dynet_main`
- `dynet_plot`

## Install

Use a virtual environment and install with the same interpreter you will run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

Install docs tooling:

```bash
python -m pip install -e ".[docs]"
```

Build distribution artifacts:

```bash
python -m pip install build
python -m build
```

Verify import:

```bash
python -c "import dynet_py, sys; print(dynet_py.__version__); print(sys.executable)"
```

Troubleshooting `ModuleNotFoundError` after install:

```bash
python3 -m pip --version
python3 -c "import sys; print(sys.executable)"
```

If `pip` points to a different Python version than the interpreter you use to run code, reinstall with `python -m pip` inside a virtualenv.

## Documentation

Comprehensive docs are in `docs/` and configured via `mkdocs.yml`.

Run docs locally:

```bash
mkdocs serve
```

Build static docs:

```bash
mkdocs build --strict
```

## Minimal usage (`rewiring_analysis`-style)

```python
import pandas as pd
from dynet_py import rewiring_analysis, rewiring_plot

el1 = pd.DataFrame({"from": ["A", "B"], "to": ["B", "C"], "weight": [1, 2]})
el2 = pd.DataFrame({"from": ["A", "C"], "to": ["C", "B"], "weight": [2, 1]})

res = rewiring_analysis({"net1": el1, "net2": el2})
fig = rewiring_plot({"net1": el1, "net2": el2}, res)
```

## Example script

Runnable example:

```bash
PYTHONPATH=src python examples/basic_api_example.py
```

This writes demo outputs under `examples/output/`.

## CLI usage

Run rewiring analysis directly from CSV edge lists:

```bash
dynet-py --edge-lists input_edges_t1.csv input_edges_t2.csv --out-dir results
```

Or use this repo's bundled multi-network format:

```bash
dynet-py --input-csv input_edges.csv --out-dir results
```

Expected input columns:
- `from`
- `to`
- `weight` (optional, defaults to `1`)

For `--input-csv`, include `network` as well.

Output files:
- `dynet_py_output.csv`
- `compare_targeting.csv`
- `dynet_py_plot.png`
- `small_multiples_plot.png`

## Minimal usage (draft API)

```python
import pandas as pd
from dynet_py import package_data, dynet_main, dynet_plot

raw = pd.DataFrame(
    {
        "src": ["A", "A", "B", "C", "C"],
        "dst": ["B", "C", "C", "D", "A"],
        "cond": ["T0", "T0", "T1", "T1", "T1"],
        "w": [1.0, 1.2, 0.5, 2.0, 1.0],
    }
)

packed = package_data(raw, source="src", target="dst", condition="cond", weight="w")
out = dynet_main(packed)
fig = dynet_plot(out, what="edges")
```
