# dynet-py

`dynet-py` is a Python package for network rewiring analysis.

It installs the Python import package `dynet_py` and the CLI command `dynet-py`.

It exposes the main API:

- `prepare_networks`
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
from dynet_py import prepare_networks, rewiring_analysis, rewiring_plot

el1 = pd.DataFrame({"from": ["A", "B"], "to": ["B", "C"], "weight": [1, 2]})
el2 = pd.DataFrame({"from": ["A", "C"], "to": ["C", "B"], "weight": [2, 1]})

networks = prepare_networks({"net1": el1, "net2": el2})
res = rewiring_analysis(networks)
fig = rewiring_plot(networks, res)
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

## Reuse prepared networks

When running several analyses or plots on the same inputs, prepare them once:

```python
from dynet_py import (
    prepare_networks, rewiring_analysis, compare_targeting,
    calculate_jaccard_indices, rewiring_plot, small_multiples_plot,
)

networks = prepare_networks({"net1": el1, "net2": el2})
res = rewiring_analysis(networks)
targeting = compare_targeting(networks)
jaccard = calculate_jaccard_indices(networks)
fig = rewiring_plot(networks, res)
small_fig = small_multiples_plot(networks, focus_node="A")
```

`prepare_networks()` returns a reusable `PreparedNetworks` collection. Every
calculation and plot in the example consumes this same collection; passing it to
`prepare_networks()` again returns the identical object without rechecking each
network or rebuilding the collection. Raw inputs remain accepted for convenience,
but passing raw inputs separately to each function repeats preparation.

Prepared inputs are snapshots: prepare the raw inputs again after editing the
original networks. The collection supports reading names and values like a
mapping, but entries cannot be added, replaced, or removed. Treat its values as
opaque. Network names and isolated nodes are retained. Duplicate directed edges
are summed; zero-weight edges (including duplicates that cancel) are absent from
structural comparisons. Jaccard compares labeled directed edges, so node ordering
and different node sets are handled correctly.

For edge-list inputs, targeting, Jaccard, and plotting operate on normalized edges
without allocating dense adjacency matrices. Rewiring builds one cached NumPy
array shaped `(networks, nodes, nodes)`, aligned to the union of node labels, and
performs its calculations over the whole array. Edge-list inputs populate this
array directly, without intermediate DataFrame matrices. The CLI automatically
reuses prepared networks.

The aligned array and its labels are also available explicitly:

```python
adjacency = networks.adjacency_tensor()  # read-only float64; created on first use
network_names = tuple(networks)         # axis 0
node_names = networks.node_names        # source axis 1 and target axis 2
```

Nodes absent from a network have zero rows and columns. Repeated calls return the
same cached array; use `adjacency.copy()` to obtain a writable copy. Structural
rewiring does not change the cached weights. Dense storage scales with the number
of networks times the square of the total node count, so this array is only
allocated when requested or needed for rewiring.

For compatibility, `format_indata` still returns a list of per-network DataFrame
matrices with their original node sets. These are copies, so editing an exported
matrix does not alter the prepared inputs.

### Validation at preparation

`prepare_networks()` validates and normalizes every raw network before it reaches
a calculation or plot:

- Network names and node labels become strings. Missing labels, duplicate matrix
  labels, and distinct labels that collide after conversion (such as `1` and
  `"1"` within one network) are rejected.
- Weights become finite `float64` values. Numeric strings are accepted. Invalid
  text, missing values, infinity, complex numbers, and datetime values raise
  `ValueError`, with the affected network's name in the message. An omitted
  weight column or graph edge weight still defaults to `1.0`.
- Adjacency matrices must be square. Labeled rows and columns must identify the
  same nodes; rows are aligned to column order once. A default row index
  (`RangeIndex(0, n)`) inherits column labels by position when the labels differ.
  Arrays and nested lists use positional node labels (`"0"`, `"1"`, ...).
- Duplicate edge weights are summed, and a sum that overflows is rejected.

This validation is stricter than earlier versions, which silently replaced some
invalid weights with zero. Calculations and plots now rely on the prepared data
types instead of repeatedly converting labels and weights. Their existing output
column formats are retained. Prepare the raw inputs again after correcting or
editing them.
