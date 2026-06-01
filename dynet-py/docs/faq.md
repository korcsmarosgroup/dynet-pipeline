# FAQ

## Why do I get matplotlib cache warnings?

In restricted environments, default cache paths may be unwritable. Set:

```bash
export MPLCONFIGDIR=/tmp/matplotlib
```

## What input format does `rewiring_analysis` need?

Each network can be:

- edge-list DataFrame with `from`, `to`, optional `weight`
- adjacency matrix (square DataFrame or ndarray)
- graph-like object with `.nodes()` and `.edges()`

## How many networks are required?

At least two networks are required for rewiring calculations.

## Can I run structure-only rewiring?

Yes. Use:

```python
rewiring_analysis(networks, structure_only=True)
```

or CLI:

```bash
dynet-py --input-csv input_edges.csv --structure-only
```
