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

## Main Options

- `--out-dir`: output directory (default `dynet_py_results`)
- `--structure-only`: compute structural rewiring only
- `--focus-node`: node to highlight in small multiples plot

## Example Commands

```bash
dynet-py --edge-lists net_t1.csv net_t2.csv --out-dir results
dynet-py --input-csv input_edges.csv --structure-only --out-dir results_structure
```

## Generated Outputs

- `dynet_py_output.csv`
- `compare_targeting.csv`
- `dynet_py_plot.png`
- `small_multiples_plot.png`
