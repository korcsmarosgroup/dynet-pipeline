# Quickstart

## API Example

```python
import pandas as pd
from dynet_py import rewiring_analysis, rewiring_plot, compare_targeting

net1 = pd.DataFrame(
    {"from": ["A", "A", "B"], "to": ["B", "C", "C"], "weight": [1.0, 2.0, 1.0]}
)
net2 = pd.DataFrame(
    {"from": ["A", "B", "C"], "to": ["C", "C", "B"], "weight": [2.0, 1.0, 1.0]}
)

networks = {"net1": net1, "net2": net2}
rewiring = rewiring_analysis(networks)
targeting = compare_targeting(networks)
fig = rewiring_plot(networks, rewiring)
```

## Run The Included Example

```bash
PYTHONPATH=src python examples/basic_api_example.py
```

Outputs are written to `examples/output/`.

## CLI Example

```bash
dynet-py --input-csv input_edges.csv --out-dir results
```
