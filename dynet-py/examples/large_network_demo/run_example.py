"""Large-network plotting example. From the dynet-py directory, run:
python examples/large_network_demo/run_example.py
"""

import os
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)

from pathlib import Path
from time import perf_counter
from importlib.metadata import version
from platform import python_version
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def display(value):
    return None

# Support a kernel started here, at the repository root, or in dynet-py.
candidates = [
    Path.cwd(),
    Path.cwd() / "dynet-py" / "examples" / "large_network_demo",
    Path.cwd() / "examples" / "large_network_demo",
]
DEMO_DIR = next((path for path in candidates if (path / "synthetic_networks.py").exists()), None)
if DEMO_DIR is None:
    raise RuntimeError("Start the notebook in its folder, the repository root, or the dynet-py directory.")
sys.path.insert(0, str(DEMO_DIR))
from synthetic_networks import make_networks
from dynet_py import prepare_networks, rewiring_analysis, rewiring_plot, small_multiples_plot

N_NODES = 2000
SEED = 20261019
OUTPUT_DIR = DEMO_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
timings = []
views = []

started = perf_counter()
raw_networks = make_networks(n_nodes=N_NODES, seed=SEED)
timings.append({"step": "generate synthetic inputs", "seconds": perf_counter() - started})

started = perf_counter()
networks = prepare_networks(raw_networks)
timings.append({"step": "prepare inputs", "seconds": perf_counter() - started})

started = perf_counter()
scores = rewiring_analysis(networks, backend="sparse")
timings.append({"step": "sparse rewiring scores", "seconds": perf_counter() - started})
scores_before_plotting = scores.copy(deep=True)
assert len(scores) == N_NODES

input_summary = pd.DataFrame([
    {"condition": name, "nodes": N_NODES, "directed_edges": len(frame)}
    for name, frame in raw_networks.items()
])
union_edges = pd.concat(raw_networks.values(), ignore_index=True)[["source", "target"]].drop_duplicates()
print(f"Directed union edges: {len(union_edges):,}")
display(input_summary)
display(scores.sort_values("rewiring", ascending=False).head(10))

# A moderately sized neighborhood is more useful than the changing hub's huge one.
candidates = scores.loc[scores["degree"].between(10, 25)].sort_values("rewiring", ascending=False)
FOCUS_NODE = candidates.iloc[0]["name"]
print("Focal node:", FOCUS_NODE)

def save_view(figure, filename, title, drawn_nodes, started):
    figure.savefig(OUTPUT_DIR / filename, dpi=140, bbox_inches="tight")
    timings.append({"step": filename, "seconds": perf_counter() - started})
    views.append({"view": title, "displayed_nodes": drawn_nodes, "file": filename})
    display(figure)
    plt.close(figure)

started = perf_counter()
figure = rewiring_plot(networks, scores, figsize=(12, 9))
figure.suptitle(f"Automatic full-network view: {N_NODES:,} nodes")
figure.tight_layout()
save_view(figure, "01_full_network_auto.png", "Full network: automatic", N_NODES, started)

started = perf_counter()
figure = rewiring_plot(networks, scores, layout="spring", labels="none", seed=SEED, figsize=(12, 9))
figure.suptitle(f"Spring full-network view: {N_NODES:,} nodes")
figure.tight_layout()
save_view(figure, "02_full_network_spring.png", "Full network: spring", N_NODES, started)

started = perf_counter()
figure = rewiring_plot(networks, scores, top_n=40, layout="spring", labels="auto", seed=SEED, figsize=(10, 8))
save_view(figure, "03_top_40_nodes.png", "Top 40 by rewiring", 40, started)
display(scores.sort_values("rewiring", ascending=False).head(40)[["name", "rewiring", "degree"]])

incident = union_edges[(union_edges["source"] == FOCUS_NODE) | (union_edges["target"] == FOCUS_NODE)]
neighbor_names = {FOCUS_NODE, *incident["source"], *incident["target"]}
neighbor_scores = scores.loc[scores["name"].isin(neighbor_names)].sort_values("rewiring", ascending=False)
focus_labels = [FOCUS_NODE, *neighbor_scores["name"].head(4)]
NEIGHBORHOOD_SIZE = min(25, len(neighbor_names))
print(f"Neighborhood around {FOCUS_NODE}: {len(neighbor_names)} union nodes, displaying {NEIGHBORHOOD_SIZE}.")

started = perf_counter()
figure = rewiring_plot(
    networks, scores, focus_node=FOCUS_NODE, radius=1, top_n=25,
    layout="spring", labels=focus_labels, seed=SEED, figsize=(10, 8),
)
save_view(figure, "04_focused_neighborhood.png", "One-hop neighborhood", NEIGHBORHOOD_SIZE, started)

started = perf_counter()
figure = small_multiples_plot(
    networks, focus_node=FOCUS_NODE, radius=1, max_nodes=25,
    layout="spring", labels=[FOCUS_NODE], seed=SEED, figsize=(16, 5),
)
save_view(figure, "05_neighborhood_comparison.png", "Three condition panels", NEIGHBORHOOD_SIZE, started)

pd.testing.assert_frame_equal(scores, scores_before_plotting)
pd.testing.assert_frame_equal(rewiring_analysis(networks, backend="sparse"), scores_before_plotting)
for name, frame in raw_networks.items():
    frame.to_csv(OUTPUT_DIR / f"{name}_edges.csv", index=False)
scores.to_csv(OUTPUT_DIR / "rewiring_scores_all_nodes.csv", index=False)
input_summary.to_csv(OUTPUT_DIR / "network_summary.csv", index=False)
view_summary = pd.DataFrame(views)
view_summary.to_csv(OUTPUT_DIR / "view_summary.csv", index=False)
timing_table = pd.DataFrame(timings)
timing_table.to_csv(OUTPUT_DIR / "illustrative_run_timings.csv", index=False)
metadata = {
    "nodes_per_condition": N_NODES,
    "edges_per_condition": {name: len(frame) for name, frame in raw_networks.items()},
    "directed_union_edges": len(union_edges),
    "conditions": list(raw_networks),
    "seed": SEED,
    "focus_node": FOCUS_NODE,
    "neighborhood_nodes": len(neighbor_names),
    "checks": "All full scores retained and unchanged after plotting",
    "software_versions": {"python": python_version(), **{name: version(name) for name in
        ("dynet-py", "numpy", "pandas", "matplotlib", "networkx", "scipy")}},
    "purpose": "Synthetic plotting demonstration; not validation or a p-value benchmark",
}
(OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
display(view_summary)
display(timing_table.round({"seconds": 3}))
print("All plotting checks passed. Files saved in the example outputs/ folder.")
