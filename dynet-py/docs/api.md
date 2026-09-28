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

### Draft API helpers

- `package_data`
- `package_data_rename`
- `package_data_remap`
- `dynet_internal`
- `dynet_main`
- `dynet_plot`

## Module Reference

::: dynet_py.core
