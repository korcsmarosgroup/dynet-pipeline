# Quickstart

## Jupyter Notebook

For an executable walkthrough with known expected results, open the
[tutorial notebook](https://github.com/korcsmarosgroup/dynet-pipeline/blob/main/dynet-py/examples/dynet_py_tutorial.ipynb).
From `dynet-py`, with your Python environment active:

```bash
python -m pip install -e . jupyterlab ipykernel
python -m jupyterlab examples/dynet_py_tutorial.ipynb
```

Select a kernel from the same environment, then **Restart Kernel and Run All
Cells**. It covers both workflows, flat tables, plotting, CSV export, aliases,
validation, and the dense/sparse backends. Its export section writes files under
`tutorial_output/` in the notebook's working directory.

## API Example

This calculates standardized per-node rewiring scores. The separate
`compare_conditions` workflow reports edge turnover and degree changes; see the
[API reference](api.md) when those are the results you need.

```python
import pandas as pd
from dynet_py import prepare_networks, rewiring_analysis, rewiring_plot, compare_targeting

net1 = pd.DataFrame(
    {"source": ["A", "A", "B"], "target": ["B", "C", "C"], "weight": [1.0, 2.0, 1.0]}
)
net2 = pd.DataFrame(
    {"source": ["A", "B", "C"], "target": ["C", "C", "B"], "weight": [2.0, 1.0, 1.0]}
)

networks = prepare_networks({"net1": net1, "net2": net2})
rewiring = rewiring_analysis(networks)
targeting = compare_targeting(networks)
fig = rewiring_plot(networks, rewiring)
```

## Condition Comparison Tables

```python
from dynet_py import prepare_condition_data, compare_conditions, plot_condition_changes

data = prepare_condition_data(pd.concat([
    net1.assign(condition="T0"), net2.assign(condition="T1"),
], ignore_index=True))
nodes = compare_conditions(data, conditions=["T0", "T1"])
print(nodes[["node", "degree_change_score"]])
edges = compare_conditions(data, conditions=["T0", "T1"], output="edges")
summary = compare_conditions(data, conditions=["T0", "T1"], output="summary")
nodes.to_csv("node_changes.csv", index=False)
fig = plot_condition_changes(summary)
```

These are degree changes and edge counts. For standardized rewiring scores, use
the first example's `rewiring_analysis` table.

## Run The Included Example

```bash
PYTHONPATH=src python examples/basic_api_example.py
```

Outputs are written to `examples/output/`.

## CLI Example

```bash
dynet-py --input-csv input_edges.csv --out-dir results
```
