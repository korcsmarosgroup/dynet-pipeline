# CLI

The package installs a command:

```bash
dynet-py --help
```

## Input Modes

Provide exactly one of:

- `--edge-lists file1.csv file2.csv ...`
- `--input-csv input_edges.csv`

### `--edge-lists` format

Each CSV must contain:

- `from`
- `to`
- `weight` (optional; defaults to 1)

### `--input-csv` format

Single CSV with:

- `network`
- `from`
- `to`
- `weight` (optional; defaults to 1)

Inputs use the same strict validation as `prepare_networks()`: invalid weights
and missing network or node labels raise an error before results are written.
Weights must be finite real numbers; numeric strings are accepted. Missing cells
in a supplied weight column are errors, while omitting that column defaults to 1.

## Main Options

- `--out-dir`: output directory (default `dynet_py_results`)
- `--structure-only`: compute structural rewiring only
- `--backend`: `auto` (default), `sparse`, or `dense`; auto selects sparse for
  at least 100,000 aligned tensor cells with at most 10% nonzero entries
- `--focus-node`: node to highlight in small multiples plot

## Example Commands

```bash
dynet-py --edge-lists net_t1.csv net_t2.csv --out-dir results
dynet-py --input-csv input_edges.csv --structure-only --out-dir results_structure
dynet-py --input-csv input_edges.csv --backend sparse --out-dir results_sparse
```

## Generated Outputs

- `dynet_py_output.csv`
- `compare_targeting.csv`
- `dynet_py_plot.png`
- `small_multiples_plot.png`
