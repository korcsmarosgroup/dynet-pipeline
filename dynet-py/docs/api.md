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
edge lists and adjacency matrices are cached when needed. The collection is a
read-only mapping of network names to opaque values. Existing raw-input calls
still work, but each separate call prepares those raw inputs again.

Preparation guarantees string names and node labels, finite `float64` weights,
and square matrices with aligned row/column labels. Numeric strings are accepted;
missing, nonnumeric, infinite, complex, or datetime weights raise `ValueError`.
An omitted weight column defaults to `1.0`. Missing labels and labels that become
ambiguous after string conversion are rejected. Matrix rows with explicit labels
are aligned to the columns; a default positional row index inherits column labels
when they differ. Errors identify the affected network. These checks run once on
raw inputs and are skipped when reusing the prepared collection.

### Draft API helpers

- `package_data`
- `package_data_rename`
- `package_data_remap`
- `dynet_internal`
- `dynet_main`
- `dynet_plot`

## Module Reference

::: dynet_py.core
