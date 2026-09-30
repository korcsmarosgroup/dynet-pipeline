# FAQ

## Does `dynet_main` run `rewiring_analysis`?

No. `dynet_main` is the historical name for `compare_conditions`, which compares
gained/lost/kept edges and changes in node degree. Its historical `rewiring_score`
column means `abs(delta_degree) + abs(delta_weight_degree)`.

Use `prepare_networks` followed by `rewiring_analysis` for standardized rewiring
scores. Use `prepare_condition_data` followed by `compare_conditions` for the
condition-comparison workflow; new results call the heuristic
`degree_change_score`. A node can change all its neighbors while keeping its
degree, so a zero degree-change score does not imply zero rewiring.

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
