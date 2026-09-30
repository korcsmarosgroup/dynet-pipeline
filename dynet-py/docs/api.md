# API Reference

## Choose a Workflow

| Goal | Input preparation | Analysis | Result | Plot |
| --- | --- | --- | --- | --- |
| Standardized rewiring scores across networks | `prepare_networks` | `rewiring_analysis` | DataFrame: `name`, `rewiring`, `degree`, `degree_corrected_rewiring` | `rewiring_plot` |
| Edge turnover and degree changes between conditions | `prepare_condition_data` | `compare_conditions` / `compare_condition_pair` | Dictionaries containing edge/node tables; node metric is `degree_change_score` | `plot_condition_changes` |

The two metrics answer different questions. Rewiring sums the sample variance of
each incident edge after dividing its weights by its mean nonzero weight across
networks, with absent edges treated as zero and self-loops counted once.
The condition-comparison heuristic is
`abs(delta_degree) + abs(delta_weight_degree)`, with both deltas calculated as
condition B minus A and weighted degree defined as the signed sum of incoming and
outgoing weights. It is **not** a standardized rewiring score.

For example, changing `A -> B, C -> D` into `A -> D, C -> B` with unit weights
preserves every node's degree. All degree-change scores are zero, but all four
standardized rewiring scores are 1. Use `rewiring_analysis` to detect that pattern.

The CLI runs the standardized-score workflow. `dynet_main` and `dynet_internal`
are historical names for condition comparisons, not orchestration or internals
of the score calculator.

## Standardized Rewiring Scores

- `PreparedNetworks`
- `prepare_networks`
- `format_indata`
- `rewiring_analysis`
- `rewiring_plot`
- `small_multiples_plot`
- `calculate_jaccard_indices`
- `compare_targeting`

Call `prepare_networks(raw_inputs)` once and reuse its `PreparedNetworks` result
with the functions above. Prepared collections pass through without reparsing;
edge coordinates and an aligned 3D NumPy adjacency array are cached when needed.
The collection is a read-only mapping of network names to opaque values. Existing
raw-input calls still work, but each separate call prepares those raw inputs again.

Preparation guarantees string names and node labels, finite `float64` weights,
and square matrices with aligned row/column labels. Numeric strings are accepted;
missing, nonnumeric, infinite, complex, or datetime weights raise `ValueError`.
An omitted weight column defaults to `1.0`. Missing labels and labels that become
ambiguous after string conversion are rejected. Matrix rows with explicit labels
are aligned to the columns; a default positional row index inherits column labels
when they differ. Errors identify the affected network. These checks run once on
raw inputs and are skipped when reusing the prepared collection.

`PreparedNetworks.adjacency_tensor()` returns a cached, read-only `float64` NumPy
array with shape `(number_of_networks, number_of_nodes, number_of_nodes)`. The
network axis follows the collection's iteration order. The source and target axes
both follow `PreparedNetworks.node_names`, a tuple containing the union of node
labels in first-seen order. Missing nodes are zero-filled. Dense rewiring operates
on this array; sparse rewiring and edge-based calculations and plots do not
require it. Use `.copy()` to get writable data. `format_indata` retains its
list-of-DataFrames return format.

`rewiring_analysis(networks, structure_only=False, *, backend="auto")` accepts:

- `sparse`: reuse cached edge coordinates and compute variance including implicit
  zero observations, without allocating a dense matrix. Storage grows with the
  observed edges and node count. No SciPy dependency is required.
- `dense`: reuse the aligned adjacency tensor.
- `auto` (default): select sparse when the tensor has at least 100,000 cells and
  at most 10% nonzero entries; otherwise select dense. Density is measured against
  the union of node labels across all networks, after duplicate aggregation.

Both backends preserve the same outputs and support structural rewiring. Sparse
variance uses a centered second pass to avoid subtracting nearly equal squared
moments. Signed-weight cancellation and overflow use a bounded-memory fallback
that preserves the established NaN/Inf behavior. This fallback may be slower.

## Condition Comparisons

- `prepare_condition_data(data, source="source", target="target", condition="condition", ...)`
  validates a single table, normalizes labels and weights, and aggregates
  duplicate edges within each condition. Missing/invalid weights are rejected;
  omitted weights default to 1. Self-loops are removed by default.
- `compare_condition_pair(data, condition_a, condition_b)` returns one dictionary
  with `comparison`, `edge_changes`, `node_changes`, and `summary`.
- `compare_conditions(data, conditions=None, pairwise="adjacent")` returns a
  dictionary with `comparisons` (a list of pair results) and `edge_counts`
  (a DataFrame). `pairwise="all"` compares every selected pair.
- `plot_condition_changes(result, what="edges")` plots gained/lost/kept counts;
  `what="nodes"` plots `degree_change_score` rankings.

Supply the desired condition order explicitly, especially for time series.
Preparation sorts grouping keys; the comparison default follows their appearance
in the prepared table, not an inferred chronological order.

Unlike the rewiring-score input format, condition tables use `source`, `target`,
`condition`, and `weight` columns. Their row membership defines edges, so even
zero-weight rows count as present. If self-loops are retained, they count twice in
total degree (once incoming and once outgoing), matching the legacy comparison.
`PreparedNetworks` and comparison dictionaries are not interchangeable inputs.

```python
from dynet_py import prepare_condition_data, compare_conditions, plot_condition_changes

data = prepare_condition_data(raw, source="src", target="dst", condition="time", weight="w")
changes = compare_conditions(data, conditions=["T0", "T1"], pairwise="adjacent")
node_changes = changes["comparisons"][0]["node_changes"]
figure = plot_condition_changes(changes, what="nodes")
```

## Compatibility Names

| Historical name | Preferred name | Compatibility behavior |
| --- | --- | --- |
| `package_data` | `prepare_condition_data` | Retains historical conversion of invalid/missing weights to zero; the new function rejects them. |
| `dynet_internal` | `compare_condition_pair` | Retains `rewiring_score` as the column name for the degree-change heuristic. |
| `dynet_main` | `compare_conditions` | Retains legacy comparison dictionaries and their `rewiring_score` column. |
| `dynet_plot` | `plot_condition_changes` | Accepts legacy result dictionaries and labels plots as degree changes. |

Existing imports continue to work without deprecation warnings. New code should
use the descriptive names and `degree_change_score`. The old score column is not
renamed in existing results, and neither condition-comparison entry point calls
`rewiring_analysis`. `package_data_rename` and `package_data_remap` remain
documented compatibility utilities for selecting columns and remapping labels.

## Module Reference

::: dynet_py.core
