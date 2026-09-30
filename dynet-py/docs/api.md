# API Reference

## Public API

### Core `rewiring_analysis`-style functions

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
edge lists and an aligned 3D NumPy adjacency array are cached when needed. The
collection is a read-only mapping of network names to opaque values. Existing raw-input calls
still work, but each separate call prepares those raw inputs again.

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
labels in first-seen order. Missing nodes are zero-filled. Rewiring operates on
this array; edge-based calculations and plots do not require it. Use `.copy()` to
get writable data. `format_indata` retains its list-of-DataFrames return format.

### Draft API helpers

- `package_data`
- `package_data_rename`
- `package_data_remap`
- `dynet_internal`
- `dynet_main`
- `dynet_plot`

## Module Reference

::: dynet_py.core
