"""Display selection and optional layouts, independent of network calculations."""

from numbers import Integral
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np


def _plot_options(layout, labels, radius, limit, seed, figsize):
    if not isinstance(layout, str) or layout not in {"auto", "circular", "spring", "kamada_kawai"}:
        raise ValueError("layout must be 'auto', 'circular', 'spring', or 'kamada_kawai'")
    if not isinstance(radius, Integral) or isinstance(radius, bool) or radius < 0:
        raise ValueError("radius must be a nonnegative integer")
    if limit is not None and (not isinstance(limit, Integral) or isinstance(limit, bool) or limit < 1):
        raise ValueError("Node limits must be positive integers or None")
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(labels, str):
        if labels not in {"auto", "all", "none"}:
            raise ValueError("labels must be 'auto', 'all', 'none', or an iterable of node IDs")
    else:
        try:
            labels = [str(node) for node in labels]
        except TypeError as exc:
            raise ValueError("labels must be 'auto', 'all', 'none', or an iterable of node IDs") from exc
    try:
        dimensions = np.asarray(figsize, dtype=float)
        if dimensions.shape != (2,) or not np.isfinite(dimensions).all() or (dimensions <= 0).any():
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError("figsize must contain two positive finite dimensions") from exc
    return labels


def _ranked_nodes(nodes: Sequence[str], ranks: Mapping[str, float]) -> list[str]:
    # Stable input order breaks ties, including nodes with missing/NaN scores.
    return sorted(nodes, key=lambda node: (np.isnan(ranks.get(node, 0.)),
                                           -ranks.get(node, 0.) if not np.isnan(ranks.get(node, 0.)) else 0.))


def _select_plot_nodes(
    nodes: Sequence[str], edges: Iterable[tuple[str, str]], ranks: Mapping[str, float],
    focus: Optional[str], radius: int, limit: Optional[int],
) -> list[str]:
    candidates = set(nodes)
    if focus is not None:
        neighbors = {node: set() for node in nodes}
        for source, target in edges:
            neighbors[source].add(target)
            neighbors[target].add(source)
        candidates = {focus}
        frontier = {focus}
        for _ in range(radius):
            frontier = {neighbor for node in frontier for neighbor in neighbors[node]} - candidates
            candidates.update(frontier)
            if not frontier:
                break
    ordered = [node for node in nodes if node in candidates]
    return _limit_plot_nodes(ordered, ranks, limit, focus)


def _limit_plot_nodes(nodes, ranks, limit, focus):
    if limit is not None and len(nodes) > limit:
        ranked = _ranked_nodes(nodes, ranks)
        selected = set(ranked[:limit])
        if focus is not None and focus not in selected:
            selected.remove(ranked[limit - 1])
            selected.add(focus)
        return [node for node in nodes if node in selected]
    return list(nodes)


def _plot_labels(
    nodes: Sequence[str], ranks: Mapping[str, float], labels: Union[str, Sequence[str]],
    focus: Optional[str] = None,
) -> list[str]:
    if isinstance(labels, str):
        if labels == "none":
            return []
        if labels == "all" or len(nodes) <= 30:
            return list(nodes)
        return [focus] if focus is not None else _ranked_nodes(nodes, ranks)[:10]
    requested = set(labels)
    return [node for node in nodes if node in requested]


def _layout_positions(nodes, edges, layout, seed):
    if not nodes:
        return {}
    auto = layout == "auto"
    if auto:
        layout = "spring" if 30 < len(nodes) <= 500 else "circular"
    if layout != "circular":
        try:
            import networkx as nx

            graph = nx.Graph()
            graph.add_nodes_from(nodes)
            graph.add_edges_from(edges)
            if layout == "spring":
                return nx.spring_layout(graph, seed=int(seed), weight=None)
            # Seed the initial positions as well, so repeated calls are reproducible.
            initial = nx.circular_layout(graph)
            return nx.kamada_kawai_layout(graph, pos=initial, weight=None)
        except ImportError as exc:
            if not auto:
                raise ImportError("This layout requires optional plotting dependencies. "
                                  "Install them with: pip install 'dynet-py[plot]'") from exc
    angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    return {node: np.array([np.cos(angle), np.sin(angle)]) for node, angle in zip(nodes, angles)}
