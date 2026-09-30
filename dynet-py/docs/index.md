# dynet-py

`dynet-py` is a Python package for dynamic network rewiring analysis.

It supports:

- rewiring metrics across multiple directed or weighted networks
- comparison helpers (`compare_targeting`, `calculate_jaccard_indices`)
- plotting functions for network rewiring summaries
- a command-line interface for CSV-based workflows

For standardized scores, start with `prepare_networks` and `rewiring_analysis`.
For gained/lost edges and degree changes, use `prepare_condition_data` and
`compare_conditions`. These are different metrics; the historical `dynet_main`
name refers only to the second workflow. See [API Reference](api.md) for the
input/output formats and compatibility names.

## Who This Is For

- Researchers comparing network structure across conditions or time points
- Analysts comparing network structure across conditions or time points
- Developers building custom rewiring pipelines on top of pandas-based data

## What You Get

- installable Python package (`pip install dynet-py`)
- Python API under `dynet_py.*`
- CLI command `dynet-py`
- test suite and validation script
- reproducible examples

Use the navigation to start with [Installation](installation.md) and [Quickstart](quickstart.md).
