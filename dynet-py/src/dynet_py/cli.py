from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .core import compare_targeting, rewiring_analysis, rewiring_plot, small_multiples_plot


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dynet-py",
        description="Run dynamic network rewiring analysis from edge-list CSV files.",
    )
    parser.add_argument(
        "--edge-lists",
        nargs="+",
        metavar="CSV",
        help="Two or more CSV files with columns: from,to[,weight].",
    )
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Single CSV with columns: network,from,to[,weight].",
    )
    parser.add_argument(
        "--out-dir",
        default="dynet_py_results",
        help="Output directory for generated CSV and PNG files.",
    )
    parser.add_argument(
        "--structure-only",
        action="store_true",
        help="Run structure-only (unweighted) rewiring analysis.",
    )
    parser.add_argument(
        "--focus-node",
        default=None,
        help="Focus node used for small multiples plot (defaults to top rewired node).",
    )
    return parser.parse_args()


def _load_edge_list(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"from", "to"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns: from,to[,weight]")
    if "weight" not in df.columns:
        df = df.copy()
        df["weight"] = 1.0
    return df[["from", "to", "weight"]]


def main() -> None:
    args = _parse_args()
    if bool(args.edge_lists) == bool(args.input_csv):
        raise ValueError("Provide exactly one of --edge-lists or --input-csv.")

    if args.input_csv:
        input_df = pd.read_csv(args.input_csv)
        required = {"network", "from", "to"}
        if not required.issubset(input_df.columns):
            raise ValueError("--input-csv must contain columns: network,from,to[,weight]")
        if "weight" not in input_df.columns:
            input_df = input_df.copy()
            input_df["weight"] = 1.0
        networks = {
            str(net): grp[["from", "to", "weight"]].reset_index(drop=True)
            for net, grp in input_df.groupby("network", sort=False)
        }
    else:
        csv_paths = [Path(p) for p in args.edge_lists]
        if len(csv_paths) < 2:
            raise ValueError("At least two edge-list CSV files are required.")
        networks = {p.stem: _load_edge_list(p) for p in csv_paths}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rewiring = rewiring_analysis(networks, structure_only=args.structure_only)
    rewiring.to_csv(out_dir / "dynet_py_output.csv", index=False)

    targeting = compare_targeting(networks)
    targeting.to_csv(out_dir / "compare_targeting.csv", index=False)

    focus_node = args.focus_node or str(rewiring.sort_values("rewiring", ascending=False)["name"].iloc[0])

    fig1 = rewiring_plot(networks, rewiring, structure_only=args.structure_only)
    fig1.savefig(out_dir / "dynet_py_plot.png", dpi=150, bbox_inches="tight")
    fig2 = small_multiples_plot(networks, focus_node=focus_node)
    fig2.savefig(out_dir / "small_multiples_plot.png", dpi=150, bbox_inches="tight")

    print(f"Wrote results to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
