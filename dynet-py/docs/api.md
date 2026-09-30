# API Reference

## Choose a Workflow

| Goal | Input preparation | Analysis | Result | Plot |
| --- | --- | --- | --- | --- |
| Standardized rewiring scores across networks | `prepare_networks` | `rewiring_analysis` | DataFrame: `name`, `rewiring`, `degree`, `degree_corrected_rewiring` | `rewiring_plot` |
| Edge turnover and degree changes between conditions | `prepare_condition_data` | `compare_conditions` / `compare_condition_pair` | DataFrame: select nodes (default), edges, or summary; node metric is `degree_change_score` | `plot_condition_changes` |

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

## Edge-List Column Names

Use `source`, `target`, and optional `weight` in both workflows. Condition tables
also have a `condition` column. `prepare_networks`, `prepare_condition_data`,
legacy `package_data`, and the CLI accept these endpoint pairs:

| Input columns | Normalized columns |
| --- | --- |
| `source`, `target` (preferred) | `source`, `target` |
| `from`, `to` | `source`, `target` |
| `src`, `dst` | `source`, `target` |

Normalization happens during preparation, leaving the caller's table unchanged.
Calculations and plots consume the prepared names. Condition-edge outputs also
use `source` and `target`, so their columns do not need renaming before preparing
separate networks for scoring. The node-result schemas remain unchanged.

Duplicate column names or multiple complete endpoint pairs are rejected, even
when their values agree. Keep only one pair, or select columns explicitly with
`prepare_condition_data(data, source="origin", target="destination")` for custom
names. Selecting only one endpoint leaves the other at its canonical name.
For network inputs, a square DataFrame with matching row/column node-label sets
is treated as an adjacency matrix, even if those labels include `source`/`target`.

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

- `prepare_condition_data(data, source=None, target=None, condition="condition", ...)`
  validates a single table, normalizes labels and weights, and aggregates
  duplicate edges within each condition. Missing/invalid weights are rejected;
  omitted weights default to 1. Self-loops are removed by default.
- `compare_condition_pair(data, condition_a, condition_b, output="nodes")`
  returns a DataFrame for one condition pair.
- `compare_conditions(data, conditions=None, pairwise="adjacent", output="nodes")`
  returns a DataFrame combining the selected pairs. `pairwise="all"` compares
  every selected pair. The pair and batch functions use the same column names.
- `plot_condition_changes(table)` infers the plot from the table: node rankings
  for node tables, gained/lost/kept counts for edge or summary tables. Select a
  pair with `comparison=0` (default), `1`, etc., in appearance order; negative
  indices work too. Explicit `what="nodes"` or `"edges"` must match the table.

### Table Outputs

All comparison tables contain `comparison`, `condition_a`, and `condition_b`.
Condition labels are ordinary values rather than parts of column names. `_a`
refers to the baseline and `_b` to the comparison condition; deltas are B minus A.
Filter or join using both condition columns: display names can coincide if
condition labels themselves contain `_vs_`.

| `output` | One row per | Additional columns |
| --- | --- | --- |
| `"nodes"` (default) | Node and condition pair | `node`; `in_degree_a/b`, `out_degree_a/b`, `degree_a/b`; `in_weight_a/b`, `out_weight_a/b`, `weight_degree_a/b`; `delta_degree`, `delta_weight_degree`, `degree_change_score` |
| `"edges"` | Edge and condition pair | `source`, `target`, `status`, `weight_a`, `weight_b`, `delta_weight` |
| `"summary"` | Condition pair | `n_edges_a`, `n_edges_b`, `n_kept`, `n_lost`, `n_gained`, `n_nodes` |

Here `a/b` denotes two separate columns, for example `degree_a` and `degree_b`.
Tables have an ordinary row index and scalar cells, so standard pandas selection,
grouping, and `to_csv(index=False)` work directly. Node rows are ranked by
`degree_change_score` within each comparison; edge rows are sorted by endpoints.

Supply the desired condition order explicitly, especially for time series.
Preparation sorts grouping keys; the comparison default follows their appearance
in the prepared table, not an inferred chronological order.

Condition tables add `condition` to the shared `source`, `target`, and `weight`
format. Their row membership defines edges, so even
zero-weight rows count as present. If self-loops are retained, they count twice in
total degree (once incoming and once outgoing), matching the legacy comparison.
`PreparedNetworks` and comparison result tables are not interchangeable inputs.

```python
from dynet_py import prepare_condition_data, compare_conditions, compare_condition_pair, plot_condition_changes

data = prepare_condition_data(raw)
changes = compare_conditions(data, conditions=["T0", "T1"], pairwise="adjacent")
print(changes[["node", "degree_change_score"]])
changes.to_csv("node_changes.csv", index=False)
figure = plot_condition_changes(changes)

edges = compare_condition_pair(data, "T0", "T1", output="edges")
summary = compare_conditions(data, output="summary")
edge_figure = plot_condition_changes(summary)
```

`rewiring_analysis`, `compare_targeting`, and `calculate_jaccard_indices` already
return DataFrames and keep their existing schemas. Plotting functions return
matplotlib Figures, and `prepare_networks` returns the reusable input collection.

## Compatibility Names

| Historical name | Preferred name | Compatibility behavior |
| --- | --- | --- |
| `package_data` | `prepare_condition_data` | Retains historical conversion of invalid/missing weights to zero; the new function rejects them. |
| `dynet_internal` | `compare_condition_pair` | Same table options; `output="legacy"` restores its historical dictionary. |
| `dynet_main` | `compare_conditions` | Same table options; `output="legacy"` restores its historical nested dictionary. |
| `dynet_plot` | `plot_condition_changes` | Accepts tables and historical dictionaries; labels plots as degree changes. |

Existing imports continue to work without deprecation warnings. New code should
use the descriptive names and `degree_change_score`. Neither condition-comparison
entry point calls `rewiring_analysis`. `package_data_rename` and
`package_data_remap` remain documented compatibility utilities for selecting
columns and remapping labels.

### Migrating Nested Results

The default return type of all four comparison entry points is now a DataFrame.
Replace `result["comparisons"][0]["node_changes"]` with direct table access:

```python
nodes = compare_conditions(data)
first_pair = nodes[(nodes["condition_a"] == "T0") & (nodes["condition_b"] == "T1")]
scores = first_pair["degree_change_score"]
```

For code that needs the earlier structure, use `output="legacy"`. Batch functions
then return `comparisons` and `edge_counts`; pair functions return `comparison`,
`edge_changes`, `node_changes`, and `summary`, including historical
condition-specific measurement columns. Plotting accepts these dictionaries as
well as the new tables, including previously saved results.

### Migrating the Historical Score Column

**The default output schema of `dynet_main` and `dynet_internal` has changed:**
their node tables now contain `degree_change_score` instead of `rewiring_score`.
The formula, values, and ranking are unchanged. Update column lookups and any CSV
consumers accordingly. This correction does not add standardized rewiring scores
to the condition-comparison output.

```python
# Correct name by default, including through the historical entry point:
from dynet_py import dynet_main

changes = dynet_main(data)
degree_changes = changes["degree_change_score"]

# Explicit compatibility option for a script requiring the old schema:
old_format = dynet_main(data, output="legacy", legacy_score_name=True)
# Its "rewiring_score" column still contains the degree-change heuristic.
```

The keyword-only `legacy_score_name` option is also available on `dynet_internal`.
It changes only the column name. Existing saved results are not modified, and
`dynet_plot` accepts both names. For standardized scores, use
`rewiring_analysis(networks)["rewiring"]`.

## Module Reference

::: dynet_py.core
