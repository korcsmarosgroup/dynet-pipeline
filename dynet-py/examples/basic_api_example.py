"""Minimal dynet-py API example.

Run from repo root:
    PYTHONPATH=src python examples/basic_api_example.py
"""

from pathlib import Path

import pandas as pd

from dynet_py import compare_targeting, rewiring_analysis, rewiring_plot, small_multiples_plot


def main() -> None:
    net1 = pd.DataFrame(
        {
            "from": ["A", "A", "B", "C"],
            "to": ["B", "C", "C", "A"],
            "weight": [1.0, 2.0, 1.0, 1.0],
        }
    )
    net2 = pd.DataFrame(
        {
            "from": ["A", "B", "C", "D"],
            "to": ["C", "C", "B", "A"],
            "weight": [2.0, 1.0, 1.0, 1.0],
        }
    )

    networks = {"net1": net1, "net2": net2}

    rewiring = rewiring_analysis(networks)
    targeting = compare_targeting(networks)

    out_dir = Path("examples/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    rewiring.to_csv(out_dir / "dynet_py_output.csv", index=False)
    targeting.to_csv(out_dir / "compare_targeting.csv", index=False)

    fig1 = rewiring_plot(networks, rewiring)
    fig1.savefig(out_dir / "dynet_py_plot.png", dpi=150, bbox_inches="tight")
    fig2 = small_multiples_plot(networks, focus_node="C")
    fig2.savefig(out_dir / "small_multiples_plot.png", dpi=150, bbox_inches="tight")

    print(f"Saved outputs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
