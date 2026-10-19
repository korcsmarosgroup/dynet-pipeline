"""Compare full-network, ranked, and neighborhood views on synthetic networks.

From the dynet-py directory, install and run with:
    python -m pip install -e ".[plot]"
    python examples/large_network_plotting.py

The figures use NetworkX spring layouts and therefore require the plot extra.
Input preparation and sparse scoring themselves do not require that extra.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dynet_py import (
    prepare_networks,
    rewiring_analysis,
    rewiring_plot,
    small_multiples_plot,
)


def synthetic_networks(seed: int = 42) -> dict[str, pd.DataFrame]:
    """Create three sparse 180-node networks with shared and changing edges."""
    rng = np.random.default_rng(seed)
    n_nodes = 180
    nodes = [f"node_{i:03d}" for i in range(n_nodes)]

    # A shared directed ring keeps every node present in every input network.
    common = {(i, (i + 1) % n_nodes) for i in range(n_nodes)}
    # Most additional shared edges stay within one of six 30-node groups.
    while len(common) < 380:
        group = int(rng.integers(6)) * 30
        source, target = group + rng.choice(30, size=2, replace=False)
        common.add((int(source), int(target)))

    networks = {}
    for condition in range(3):
        edges = set(common)
        while len(edges) < 460:
            source, target = rng.choice(n_nodes, size=2, replace=False)
            edges.add((int(source), int(target)))
        # Change the partners of one hub to make a useful focal neighborhood.
        for target in rng.choice(np.arange(1, n_nodes), size=25, replace=False):
            edges.add((0, int(target)))
        ordered = sorted(edges)
        networks[f"condition_{condition}"] = pd.DataFrame(
            {
                "source": [nodes[source] for source, _ in ordered],
                "target": [nodes[target] for _, target in ordered],
                "weight": rng.uniform(0.5, 2.0, size=len(ordered)),
            }
        )
    return networks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "large_network_output",
        help="directory for input edge lists, scores, and PNG figures",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_networks = synthetic_networks()
    for name, edges in raw_networks.items():
        edges.to_csv(args.out_dir / f"{name}_edges.csv", index=False)

    prepared = prepare_networks(raw_networks)
    scores = rewiring_analysis(prepared, backend="sparse")
    scores.to_csv(args.out_dir / "rewiring_scores.csv", index=False)
    top_labels = scores.sort_values("rewiring", ascending=False)["name"].head(10)

    figures = {
        # Default labels stay sparse even though the full graph is displayed.
        "full_network": rewiring_plot(prepared, scores),
        "top_40_nodes": rewiring_plot(
            prepared, scores, top_n=40, layout="spring", labels=top_labels, seed=0,
        ),
        "focused_neighborhood": rewiring_plot(
            prepared, scores, focus_node="node_030", radius=1, top_n=40,
            layout="spring", labels=["node_030", *top_labels], seed=0,
        ),
        # The changing hub has a large neighborhood, so this view uses the cap.
        # Shared positions make condition-to-condition changes easier to follow.
        "focused_small_multiples": small_multiples_plot(
            prepared, focus_node="node_000", radius=2, max_nodes=40,
            layout="spring", seed=0, figsize=(15, 5),
        ),
    }
    for name, figure in figures.items():
        figure.savefig(args.out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(figure)

    print(f"Saved {len(figures)} figures, input edge lists, and full scores to {args.out_dir.resolve()}")
    print("Figure filters do not change the scores calculated for all 180 nodes.")


if __name__ == "__main__":
    main()
