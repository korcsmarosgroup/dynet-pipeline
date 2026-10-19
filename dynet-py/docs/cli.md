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

- `source`
- `target`
- `weight` (optional; defaults to 1)

### `--input-csv` format

Single CSV with:

- `network`
- `source`
- `target`
- `weight` (optional; defaults to 1)

Both modes also accept `from`/`to` or `src`/`dst` as endpoint aliases. The parser
normalizes them to `source`/`target`; multiple endpoint pairs in one CSV are
rejected as ambiguous. Input files using the earlier names remain supported.

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

## Plot Options

Network figures choose their layout and labels automatically. For 31–500
displayed nodes, install the optional layouts to enable automatic spring
positions:

```bash
python -m pip install -e ".[plot]"
```

Circular plotting remains available without this extra. `auto` uses circular
positions for at most 30 and more than 500 displayed nodes, and falls back to
circular if the optional layout dependencies are absent.

| Option | Applies to | Behavior |
| --- | --- | --- |
| `--plot-layout` | Both figures | `auto` (default), `circular`, `spring`, or `kamada_kawai`. Explicit spring and Kamada–Kawai require the plotting extra. |
| `--plot-labels` | Both figures | `auto` (default), `all`, or `none`. Automatic labels show all nodes up to 30 displayed nodes; larger rewiring plots label the top 10 scores while retaining a focal label within that cap, and larger small-multiple panels label only the focal node. |
| `--plot-top-n N` | Rewiring figure | Draw at most N nodes ranked by rewiring score; N must be positive. |
| `--plot-focus-node NODE` | Rewiring figure | Restrict the figure to NODE's neighborhood. If a top-N cap is set, retain this node. |
| `--plot-radius N` | Focal neighborhoods in both figures | Follow N hops along incoming or outgoing edges in the network union (default 1). Zero shows only the focal node. |
| `--panel-max-nodes N` | Small multiples | Draw at most N nodes ranked by union degree, retaining the focal node; N must be positive. |
| `--plot-seed N` | Both layouts | Seed for reproducible spring positions (default 0). |

`--focus-node` selects the small-multiple focal node; `--plot-focus-node` selects
the rewiring figure's focal node. Panels share positions across networks, with
each panel showing its own directed edges between the selected nodes. Figure
filters do not change the calculated scores or CSV outputs. Filtered figures
report their displayed node counts. Rewiring colors keep the full score range
across filtered and full figures.

## Example Commands

```bash
dynet-py --edge-lists net_t1.csv net_t2.csv --out-dir results
dynet-py --input-csv input_edges.csv --structure-only --out-dir results_structure
dynet-py --input-csv input_edges.csv --backend sparse --out-dir results_sparse
dynet-py --input-csv input_edges.csv --plot-layout spring --plot-top-n 50 --plot-labels auto --panel-max-nodes 30 --out-dir results_filtered
dynet-py --input-csv input_edges.csv --focus-node GENE1 --plot-focus-node GENE1 --plot-radius 2 --plot-top-n 40 --panel-max-nodes 40 --out-dir results_neighborhood
```

## Generated Outputs

- `dynet_py_output.csv`
- `compare_targeting.csv`
- `dynet_py_plot.png`
- `small_multiples_plot.png`
