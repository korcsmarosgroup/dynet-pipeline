# Tutorial: Basic Workflow

This tutorial shows the full flow from edge-list data to rewiring outputs.

## 1. Prepare Data

Your network tables must include:

- `from`
- `to`
- `weight` (optional)

## 2. Run Analysis

```python
import pandas as pd
from dynet_py import rewiring_analysis

net1 = pd.read_csv("net1.csv")
net2 = pd.read_csv("net2.csv")
result = rewiring_analysis({"net1": net1, "net2": net2})
print(result.head())
```

## 3. Compare Targeting

```python
from dynet_py import compare_targeting

targeting = compare_targeting({"net1": net1, "net2": net2})
print(targeting.head())
```

## 4. Plot Results

```python
from dynet_py import rewiring_plot, small_multiples_plot

fig1 = rewiring_plot({"net1": net1, "net2": net2}, result)
fig1.savefig("dynet_py_plot.png", dpi=150, bbox_inches="tight")

fig2 = small_multiples_plot({"net1": net1, "net2": net2}, focus_node="A")
fig2.savefig("small_multiples_plot.png", dpi=150, bbox_inches="tight")
```

## 5. Save Tabular Outputs

```python
result.to_csv("dynet_py_output.csv", index=False)
targeting.to_csv("compare_targeting.csv", index=False)
```
