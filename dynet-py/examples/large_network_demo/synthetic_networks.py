"""Deterministic illustrative inputs for a local large-network plotting demo.

The default data contain three directed, positively weighted networks, each
with 2,000 nodes and exactly 8,000 edges.  They share a sparse background of
communities and a ring that keeps every node present.  A small subnetwork and
one hub deliberately change between conditions so that filtered views have
something visible to explore.  These are demonstration data, not a validation
suite, biological model, or performance benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_networks(
    n_nodes: int = 2000,
    seed: int = 20261019,
) -> dict[str, pd.DataFrame]:
    """Return three reproducible synthetic edge tables for plotting examples.

    Args:
        n_nodes: Number of nodes in every condition; must be at least 60.
        seed: NumPy random-generator seed.  Reusing it gives identical tables.

    Returns:
        ``condition_1``, ``condition_2``, and ``condition_3`` tables with
        canonical ``source``, ``target``, and ``weight`` columns.  Every table
        contains exactly ``4 * n_nodes`` distinct edges, finite positive
        weights, all node labels, and no self-loops.  Labels are ``node_0000``
        onward, with a four-digit minimum width.  ``node_0000`` is the changing
        hub; the next few dozen labels form the deliberately changing module.

    Notes:
        Community membership controls most background connections.  Background
        weights receive only small changes across conditions, while the module,
        hub, and a minority of other links change their partners.  The data are
        designed to illustrate plotting controls, not establish correctness or
        estimate runtime on biological datasets.
    """
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, (int, np.integer)):
        raise TypeError("n_nodes must be an integer of at least 60.")
    if n_nodes < 60:
        raise ValueError("n_nodes must be at least 60.")

    n_nodes = int(n_nodes)
    rng = np.random.default_rng(seed)
    labels = [f"node_{index:04d}" for index in range(n_nodes)]
    edge_count = 4 * n_nodes
    common_count = int(3.2 * n_nodes)
    group_count = max(2, int(np.ceil(n_nodes / 100)))
    groups = np.array_split(np.arange(n_nodes), group_count)
    module_size = min(48, max(12, n_nodes // 10))
    module_nodes = np.arange(1, module_size + 1)

    # The ring provides all labels in every input and connects the communities.
    common: set[tuple[int, int]] = {
        (source, (source + 1) % n_nodes) for source in range(n_nodes)
    }
    while len(common) < common_count:
        # Most background edges stay within a community, with occasional links
        # between communities; direction is retained throughout the example.
        if rng.random() < 0.9:
            group = groups[int(rng.integers(group_count))]
            source, target = rng.choice(group, size=2, replace=False)
        else:
            source, target = rng.choice(n_nodes, size=2, replace=False)
        common.add((int(source), int(target)))

    ordered_common = sorted(common)
    base_weights = rng.uniform(0.5, 1.5, size=len(ordered_common))
    module_pool = [
        (int(source), int(target))
        for source in module_nodes
        for target in module_nodes
        if source != target and (int(source), int(target)) not in common
    ]
    variable_count = edge_count - common_count
    module_count = min(400, variable_count // 3, len(module_pool))
    # Hub connections extend outside the changing module so that its broad
    # neighborhood illustrates why a displayed-node cap can be useful.
    hub_pool = [
        (0, target)
        for target in range(module_size + 1, n_nodes)
        if (0, target) not in common
    ]
    hub_count = min(90, max(4, n_nodes // 20), len(hub_pool))

    networks: dict[str, pd.DataFrame] = {}
    for condition in range(3):
        weights = {
            edge: float(weight * rng.uniform(0.98, 1.02))
            for edge, weight in zip(ordered_common, base_weights)
        }
        for index in rng.choice(len(module_pool), size=module_count, replace=False):
            weights[module_pool[int(index)]] = float(rng.uniform(1.0, 2.5))
        for index in rng.choice(len(hub_pool), size=hub_count, replace=False):
            weights[hub_pool[int(index)]] = float(rng.uniform(0.8, 1.8))

        # The remaining changing links are dispersed across the whole network.
        while len(weights) < edge_count:
            source, target = rng.choice(n_nodes, size=2, replace=False)
            edge = (int(source), int(target))
            if edge not in weights:
                weights[edge] = float(rng.uniform(0.5, 1.5))

        ordered_edges = sorted(weights)
        networks[f"condition_{condition + 1}"] = pd.DataFrame(
            {
                "source": [labels[source] for source, _ in ordered_edges],
                "target": [labels[target] for _, target in ordered_edges],
                "weight": [weights[edge] for edge in ordered_edges],
            }
        )
    return networks
