from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, Iterator, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _is_edge_list_df(df: pd.DataFrame) -> bool:
    return {"from", "to"}.issubset(df.columns)


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _coerce_named_inputs(
    input_list: Union[Mapping[str, object], Sequence[object]],
) -> Tuple[list[object], list[str]]:
    if isinstance(input_list, Mapping):
        names = _normalize_labels(input_list.keys(), "Network names")
        values = list(input_list.values())
        return values, names

    if isinstance(input_list, (str, bytes)):
        raise ValueError("Input must be a sequence/mapping of matrices, edge lists, or graphs.")

    try:
        values = list(input_list)
    except TypeError as exc:
        raise ValueError("Input must be a sequence/mapping of matrices, edge lists, or graphs.") from exc
    names = [str(i + 1) for i in range(len(values))]
    return values, names


def _normalize_labels(labels: Iterable[object], description: str, *, unique: bool = True) -> list[str]:
    values = pd.Index(list(labels), dtype=object, tupleize_cols=False)
    if values.isna().any():
        raise ValueError(f"{description} must not contain missing values.")
    names = [str(value) for value in values]
    # Repeated endpoints are valid, but distinct IDs such as 1 and "1" must not
    # silently merge when converted to the canonical string representation.
    distinct = values if unique else values.unique()
    if len({str(value) for value in distinct}) != len(distinct):
        raise ValueError(f"{description} must be unique after conversion to strings.")
    return names


def _normalize_weights(values: object) -> np.ndarray:
    """Own finite, real float64 weights; never silently discard bad values."""
    message = "Weights must be finite real numbers (numeric strings are accepted)."
    try:
        raw = np.asarray(values)
        unsupported_objects = raw.dtype.kind == "O" and any(
            isinstance(value, (complex, np.complexfloating, np.datetime64, np.timedelta64))
            for value in raw.flat
        )
        if np.iscomplexobj(raw) or raw.dtype.kind in "mM" or unsupported_objects:
            raise ValueError(message)
        weights = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(weights).all():
        raise ValueError(message)
    return weights


class _PreparedNetwork:
    """Owned string node labels and finite float64 weights with lazy caches."""

    def __init__(
        self, nodes: Iterable[str], edges: Optional[pd.DataFrame] = None,
        adjacency: Optional[pd.DataFrame] = None,
    ):
        self.nodes = list(nodes)
        self._edges = edges
        self._adjacency = adjacency

    def edges(self) -> pd.DataFrame:
        if self._edges is None:
            self._edges = _indata_to_edgelist(self._adjacency)
        return self._edges

    def adjacency(self) -> pd.DataFrame:
        if self._adjacency is None:
            adj = pd.DataFrame(0.0, index=self.nodes, columns=self.nodes)
            for src, dst, weight in self._edges.itertuples(index=False, name=None):
                adj.loc[src, dst] = weight
            self._adjacency = adj
        return self._adjacency


def _prepare_edgelist(el: pd.DataFrame) -> _PreparedNetwork:
    if not el.columns.is_unique:
        raise ValueError("Edge-list column names must be unique.")
    labels = _normalize_labels(list(el["from"]) + list(el["to"]), "Node labels", unique=False)
    work = pd.DataFrame({"from": labels[:len(el)], "to": labels[len(el):]}, dtype=object)
    work["weight"] = (
        _normalize_weights(el["weight"])
        if "weight" in el.columns else 1.0
    )
    nodes = _ordered_unique(labels)
    grouped = work.groupby(["from", "to"], as_index=False)["weight"].sum()
    if not np.isfinite(grouped["weight"].to_numpy()).all():
        raise ValueError("Summed edge weights must be finite.")
    grouped = grouped[grouped["weight"] != 0].copy()
    # Match matrix row-major edge order without allocating a dense matrix.
    order = {node: i for i, node in enumerate(nodes)}
    grouped = grouped.sort_values(["from", "to"], key=lambda col: col.map(order)).reset_index(drop=True)
    return _PreparedNetwork(nodes, edges=grouped)


def _prepare_network(item: object) -> _PreparedNetwork:
    if isinstance(item, _PreparedNetwork):
        return item
    if isinstance(item, pd.DataFrame) and _is_edge_list_df(item):
        return _prepare_edgelist(item)
    adj = _coerce_to_adjacency_matrix(item)
    return _PreparedNetwork(adj.index, adjacency=adj)


class _SparseEdges(NamedTuple):
    """Union-edge coordinates and nonzero observations in network order."""

    sources: np.ndarray
    targets: np.ndarray
    groups: np.ndarray
    weights: np.ndarray
    counts: np.ndarray
    offsets: np.ndarray


class PreparedNetworks(Mapping[str, _PreparedNetwork]):
    """Validated networks with lazy sparse coordinates and an adjacency tensor.

    Usually created with :func:`prepare_networks`. Network names can be inspected
    like a mapping, but entries cannot be replaced or removed. Treat the values
    as opaque and pass this collection directly to analysis and plotting functions.
    """

    def __init__(self, input_list: Union[Mapping[str, object], Sequence[object]]):
        items, names = _coerce_named_inputs(input_list)
        self._networks = {}
        for name, item in zip(names, items):
            try:
                self._networks[name] = _prepare_network(item)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Network {name!r}: {exc}") from exc
        self._node_names = tuple(_ordered_unique(
            node for net in self._networks.values() for node in net.nodes
        ))
        self._adjacency_tensor: Optional[np.ndarray] = None
        self._sparse_edges: Optional[_SparseEdges] = None
        self._auto_backend: Optional[str] = None

    def __getitem__(self, name: str) -> _PreparedNetwork:
        return self._networks[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._networks)

    def __len__(self) -> int:
        return len(self._networks)

    @property
    def node_names(self) -> tuple[str, ...]:
        """Node labels in the order of both node axes of the adjacency tensor."""
        return self._node_names

    def adjacency_tensor(self) -> np.ndarray:
        """Return cached read-only float64 data shaped (networks, nodes, nodes).

        Axis 0 follows this collection's iteration order; axes 1 (source) and 2
        (target) follow ``node_names``. Nodes absent from a network have zero rows
        and columns. Use ``.copy()`` if a writable array is needed.
        """
        if self._adjacency_tensor is None:
            nodes = pd.Index(self.node_names)
            tensor = np.zeros((len(self), len(nodes), len(nodes)), dtype=np.float64)
            for i, network in enumerate(self.values()):
                if network._adjacency is not None:
                    positions = nodes.get_indexer(network.nodes)
                    tensor[i][np.ix_(positions, positions)] = network._adjacency.to_numpy()
                else:
                    # Populate the common node axes directly from normalized
                    # edges, without building intermediate per-network matrices.
                    edges = network.edges()
                    rows = nodes.get_indexer(edges["from"])
                    columns = nodes.get_indexer(edges["to"])
                    tensor[i, rows, columns] = edges["weight"].to_numpy()
            tensor.setflags(write=False)
            self._adjacency_tensor = tensor
        return self._adjacency_tensor

    def _edge_coordinates(self) -> _SparseEdges:
        if self._sparse_edges is None:
            nodes = pd.Index(self.node_names)
            edge_ids, weights, lengths = [], [], []
            for network in self.values():
                edges = network.edges()
                edge_ids.append(nodes.get_indexer(edges["from"]) * len(nodes)
                                + nodes.get_indexer(edges["to"]))
                weights.append(edges["weight"].to_numpy())
                lengths.append(len(edges))
            ids, groups = np.unique(np.concatenate(edge_ids), return_inverse=True)
            sources, targets = np.divmod(ids, max(1, len(nodes)))
            self._sparse_edges = _SparseEdges(
                sources, targets, groups, np.concatenate(weights),
                np.bincount(groups, minlength=len(ids)),
                np.concatenate(([0], np.cumsum(lengths))),
            )
            for array in self._sparse_edges:
                array.setflags(write=False)
        return self._sparse_edges

    def _rewiring_backend(self) -> str:
        if self._auto_backend is None:
            cells = len(self) * len(self.node_names) ** 2
            nonzero = sum(len(net._edges) if net._edges is not None
                          else np.count_nonzero(net._adjacency.to_numpy())
                          for net in self.values())
            # Keep tiny inputs on the existing dense path. Sparse coordinates
            # can use more memory than a dense tensor at high density.
            self._auto_backend = "sparse" if cells >= 100000 and nonzero <= 0.1 * cells else "dense"
        return self._auto_backend


def prepare_networks(input_list: Union[Mapping[str, object], Sequence[object]]) -> PreparedNetworks:
    """Snapshot inputs once for reuse across analysis and plotting functions.

    Retains network names and isolated nodes, aggregates duplicate edges, and
    normalizes node labels to strings and weights to finite float64 values.
    Invalid weights, missing or ambiguous labels, and malformed matrices raise
    ValueError here, before calculation or plotting. Builds the aligned adjacency
    tensor only when needed. An existing PreparedNetworks collection is returned
    unchanged, without inspecting or normalizing its networks again. Call this on
    the raw inputs again after editing them to create a new snapshot.
    """
    if isinstance(input_list, PreparedNetworks):
        return input_list
    return PreparedNetworks(input_list)


def _is_graph_like(item: object) -> bool:
    return hasattr(item, "nodes") and hasattr(item, "edges")


def _graph_to_adjacency(graph: object) -> pd.DataFrame:
    raw_nodes = list(graph.nodes())
    node_names = _normalize_labels(raw_nodes, "Node labels")
    labels = dict(zip(raw_nodes, node_names))
    edges = list(graph.edges(data=True))
    weights = _normalize_weights([
        data.get("weight", 1.0) if isinstance(data, Mapping) else 1.0
        for _, _, data in edges
    ])
    adj = pd.DataFrame(0.0, index=node_names, columns=node_names)
    directed = getattr(graph, "is_directed", lambda: True)()
    for (u, v, _), weight in zip(edges, weights):
        if u not in labels or v not in labels:
            raise ValueError("Graph edge endpoints must be present in the graph's nodes.")
        src, dst = labels[u], labels[v]
        adj.loc[src, dst] += weight
        if not directed:
            adj.loc[dst, src] += weight
    if not np.isfinite(adj.to_numpy()).all():
        raise ValueError("Summed edge weights must be finite.")
    return adj


def _coerce_to_adjacency_matrix(item: object) -> pd.DataFrame:
    if isinstance(item, pd.DataFrame):
        if item.shape[0] != item.shape[1]:
            raise ValueError("Adjacency matrix data frames must be square.")
        columns = _normalize_labels(item.columns, "Matrix column labels")
        rows = _normalize_labels(item.index, "Matrix row labels")
        if set(rows) != set(columns):
            if isinstance(item.index, pd.RangeIndex) and item.index.equals(pd.RangeIndex(len(item))):
                # An unlabeled row axis inherits the column labels by position.
                rows = columns
            else:
                raise ValueError("Adjacency matrix row and column labels must name the same nodes.")
        out = pd.DataFrame(_normalize_weights(item), index=rows, columns=columns)
        return out.reindex(index=columns)

    if isinstance(item, (np.ndarray, list, tuple)):
        arr = _normalize_weights(item)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("Adjacency matrices must be 2D square matrices.")
        labels = [str(i) for i in range(arr.shape[0])]
        return pd.DataFrame(arr, index=labels, columns=labels)

    if _is_graph_like(item):
        return _graph_to_adjacency(item)

    raise ValueError("Input must be a matrix, edge list data frame, or networkx graph object.")


def format_indata(
    input_list: Union[Mapping[str, object], Sequence[object]],
) -> list[pd.DataFrame]:
    return [network.adjacency().copy() for network in prepare_networks(input_list).values()]


def _indata_to_edgelist(ingraph: pd.DataFrame) -> pd.DataFrame:
    arr = ingraph.to_numpy()
    rows, cols = np.where(arr != 0)
    return pd.DataFrame(
        {
            "from": ingraph.index.to_numpy()[rows],
            "to": ingraph.columns.to_numpy()[cols],
            "weight": arr[rows, cols],
        }
    )


def rewiring_analysis(
    matrix_list: Union[Mapping[str, object], Sequence[object]],
    structure_only: bool = False,
    *,
    backend: str = "auto",
) -> pd.DataFrame:
    """Calculate rewiring using sparse edges or an aligned dense NumPy tensor.

    ``auto`` selects sparse storage at <=10% density and at least 100,000 tensor
    cells; otherwise it selects dense storage. Explicit ``sparse`` and ``dense``
    choices override selection. Sparse calculation includes absent edges as zeros
    without storing them, and preserves the same output columns and node order.
    """
    if backend not in {"auto", "sparse", "dense"}:
        raise ValueError("backend must be 'auto', 'sparse', or 'dense'")
    networks = prepare_networks(matrix_list)
    if len(networks) < 2:
        raise ValueError("rewiring_analysis requires at least two input networks.")
    if backend == "auto":
        backend = networks._rewiring_backend()
    if backend == "sparse":
        rewiring, degree = _sparse_rewiring(networks, structure_only)
    else:
        rewiring, degree = _dense_rewiring(networks, structure_only)
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = rewiring / degree
    return pd.DataFrame({
        "name": pd.Index(networks.node_names),
        "rewiring": rewiring,
        "degree": degree,
        "degree_corrected_rewiring": corrected,
    })


def _dense_rewiring(networks: PreparedNetworks, structure_only: bool) -> tuple[np.ndarray, np.ndarray]:
    tensor = networks.adjacency_tensor()
    matrices = tensor != 0 if structure_only else tensor
    counts = np.count_nonzero(matrices, axis=0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        non_zero_mean = np.divide(
            matrices.sum(axis=0), counts,
            out=np.zeros(matrices.shape[1:], dtype=np.float64), where=counts != 0,
        )
        non_zero_mean[~np.isfinite(non_zero_mean)] = 0.0
        deviations = matrices / non_zero_mean
        deviations[np.isnan(deviations)] = 0.0
        deviations -= deviations.mean(axis=0)
        np.square(deviations, out=deviations)

        # Match the established skip-NaN row/column sums, but preserve undefined
        # diagonal contributions (possible when signed edge weights cancel).
        diagonal = np.diagonal(deviations, axis1=1, axis2=2).copy()
        deviations[np.isnan(deviations)] = 0.0
        distances = deviations.sum(axis=1) + deviations.sum(axis=2) - diagonal
        rewiring = distances.sum(axis=0) / (len(networks) - 1)

        union = np.any(matrices > 0, axis=0)
        degree = (union.sum(axis=0) + union.sum(axis=1) - np.diag(union)).astype(float)
    return rewiring, degree


def _incident_sums(edges: _SparseEdges, values: np.ndarray, node_count: int) -> np.ndarray:
    loops = edges.sources == edges.targets
    return (np.bincount(edges.sources, weights=values, minlength=node_count)
            + np.bincount(edges.targets, weights=values, minlength=node_count)
            - np.bincount(edges.sources[loops], weights=values[loops], minlength=node_count))


def _sparse_rewiring(networks: PreparedNetworks, structure_only: bool) -> tuple[np.ndarray, np.ndarray]:
    edges = networks._edge_coordinates()
    count = len(networks)
    node_count = len(networks.node_names)
    if not len(edges.counts):
        return np.zeros(node_count), np.zeros(node_count)
    missing = count - edges.counts
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if structure_only:
            centers = edges.counts / count
            edge_variance = edges.counts * (1.0 - centers) ** 2 + missing * centers**2
            positive = np.ones(len(edges.counts), dtype=bool)
        else:
            means = np.bincount(edges.groups, weights=edges.weights,
                                minlength=len(edges.counts)) / edges.counts
            means[~np.isfinite(means)] = 0.0
            standardized = edges.weights / means[edges.groups]
            centers = np.bincount(edges.groups, weights=standardized,
                                  minlength=len(edges.counts)) / count
            # Center before squaring to avoid cancellation from E[x²] - E[x]².
            squared = (standardized - centers[edges.groups]) ** 2
            edge_variance = np.bincount(edges.groups, weights=squared,
                                        minlength=len(edges.counts)) + missing * centers**2
            positive = np.bincount(edges.groups, weights=edges.weights > 0,
                                   minlength=len(edges.counts)) > 0

        totals = _incident_sums(edges, edge_variance, node_count)
        if not structure_only and not np.isfinite(totals).all():
            # Signed cancellation and overflow have established NaN/Inf behavior.
            # Handle those rare cases one network at a time, still without N² data.
            totals = _sparse_nonfinite_totals(edges, standardized, centers, node_count)
        rewiring = totals / (count - 1)
        degree = _incident_sums(edges, positive, node_count)
    return rewiring, degree


def _sparse_nonfinite_totals(
    edges: _SparseEdges, standardized: np.ndarray, centers: np.ndarray, node_count: int,
) -> np.ndarray:
    totals = np.zeros(node_count)
    loops = edges.sources == edges.targets
    diagonal = np.zeros(node_count)
    for start, stop in zip(edges.offsets[:-1], edges.offsets[1:]):
        squared = centers**2  # Every absent observation is zero before centering.
        groups = edges.groups[start:stop]
        squared[groups] = (standardized[start:stop] - centers[groups]) ** 2
        diagonal[edges.sources[loops]] = squared[loops]
        squared[np.isnan(squared)] = 0.0
        totals += (np.bincount(edges.sources, weights=squared, minlength=node_count)
                   + np.bincount(edges.targets, weights=squared, minlength=node_count) - diagonal)
    return totals


def rewiring_plot(
    matrix_list: Union[Mapping[str, object], Sequence[object]],
    output_dataframe: pd.DataFrame,
    structure_only: bool = False,
):
    networks = prepare_networks(matrix_list)
    node_names = networks.node_names
    node_order = {node: i for i, node in enumerate(node_names)}
    union_edges = set()
    for net in networks.values():
        for src, dst, weight in net.edges().itertuples(index=False, name=None):
            if structure_only or weight > 0:
                union_edges.add(tuple(sorted((src, dst), key=node_order.__getitem__)))
    edges = sorted(union_edges, key=lambda edge: (node_order[edge[0]], node_order[edge[1]]))
    attr = output_dataframe.set_index("name")
    rewiring = np.array([attr["rewiring"].get(n, 0.0) for n in node_names], dtype=float)
    degree = np.array([attr["degree"].get(n, 0.0) for n in node_names], dtype=float)
    sizes = 300 + 1200 * (degree / degree.max() if degree.max() > 0 else degree + 1)
    n = max(1, len(node_names))
    angles = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
    pos = {node_names[i]: (np.cos(angles[i]), np.sin(angles[i])) for i in range(n)}
    fig, ax = plt.subplots(figsize=(9, 7))
    for src, dst in edges:
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        ax.plot([x1, x2], [y1, y2], color="gray", linewidth=1.0, alpha=0.5, zorder=1)
    scatter = ax.scatter(
        [pos[nm][0] for nm in node_names],
        [pos[nm][1] for nm in node_names],
        s=sizes,
        c=rewiring,
        cmap="Reds",
        edgecolors="black",
        linewidths=0.8,
        zorder=2,
    )
    for nm in node_names:
        ax.text(pos[nm][0], pos[nm][1], nm, fontsize=10, fontweight="bold", ha="center", va="center", zorder=3)
    fig.colorbar(scatter, ax=ax, label="rewiring")
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def small_multiples_plot(
    input_list: Union[Mapping[str, object], Sequence[object]],
    focus_node: str,
    mode: str = "focus_only",
):
    networks = prepare_networks(input_list)
    focus_node = str(focus_node)

    edges_by_network: list[pd.DataFrame] = []
    if mode not in {"focus_only", "r_compat"}:
        raise ValueError("mode must be 'focus_only' or 'r_compat'")

    for name, network in networks.items():
        el = network.edges()[["from", "to"]].copy()
        if mode == "focus_only":
            el = el[(el["from"] == focus_node) | (el["to"] == focus_node)]
        el = el.copy()
        el["id"] = name
        edges_by_network.append(el)

    all_edges = pd.concat(edges_by_network, ignore_index=True) if edges_by_network else pd.DataFrame()
    plot_ids = list(networks)

    n = max(1, len(plot_ids))
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for idx, network_id in enumerate(plot_ids):
        ax = axes_flat[idx]
        subset = all_edges[all_edges["id"] == network_id] if not all_edges.empty else pd.DataFrame()
        if subset.empty:
            nodes = [focus_node]
            edges = []
        else:
            nodes = _ordered_unique(
                [focus_node] + subset["from"].tolist() + subset["to"].tolist()
            )
            edges = list(subset[["from", "to"]].itertuples(index=False, name=None))

        m = max(1, len(nodes))
        angles = np.linspace(0, 2 * np.pi, num=m, endpoint=False)
        pos = {nodes[i]: (np.cos(angles[i]), np.sin(angles[i])) for i in range(m)}
        for src, dst in edges:
            x1, y1 = pos[src]
            x2, y2 = pos[dst]
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="gray", alpha=0.4, lw=1.0),
            )
        colors = ["#d62728" if node == focus_node else "#1f77b4" for node in nodes]
        ax.scatter([pos[nm][0] for nm in nodes], [pos[nm][1] for nm in nodes], c=colors, s=400, zorder=2)
        for nm in nodes:
            ax.text(pos[nm][0], pos[nm][1], nm, fontsize=9, fontweight="bold", ha="center", va="center", zorder=3)
        ax.set_title(network_id)
        ax.set_axis_off()
        ax.set_aspect("equal")

    for idx in range(len(plot_ids), len(axes_flat)):
        axes_flat[idx].set_axis_off()

    fig.tight_layout()
    return fig


def calculate_jaccard_indices(
    networks: Union[Mapping[str, object], Sequence[object]],
) -> pd.DataFrame:
    prepared = prepare_networks(networks)
    names = list(prepared)
    edge_sets = [set(net.edges()[["from", "to"]].itertuples(index=False, name=None))
                 for net in prepared.values()]
    n = len(edge_sets)
    jaccard_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            union = edge_sets[i] | edge_sets[j]
            val = len(edge_sets[i] & edge_sets[j]) / len(union) if union else 1.0
            jaccard_matrix[i, j] = val
            jaccard_matrix[j, i] = val
    return pd.DataFrame(jaccard_matrix, index=names, columns=names)


def compare_targeting(
    input_list: Union[Mapping[str, object], Sequence[object]],
) -> pd.DataFrame:
    formatted = prepare_networks(input_list)

    targeting_frames: list[pd.DataFrame] = []
    for i, net in enumerate(formatted.values(), start=1):
        in_targeting = net.edges().groupby("to")["weight"].sum().reindex(net.nodes, fill_value=0.0)
        frame = pd.DataFrame({"name": in_targeting.index, "targeting": in_targeting.to_numpy()})
        frame["network_id"] = i
        targeting_frames.append(frame[["name", "targeting", "network_id"]])

    if not targeting_frames:
        return pd.DataFrame(
            columns=[
                "name",
                "compared_networks",
                "targetingNet1",
                "targetingNet2",
                "deltaTargeting",
                "log2TargetingFC",
            ]
        )

    all_targeting = pd.concat(targeting_frames, ignore_index=True)
    network_ids = sorted(all_targeting["network_id"].unique())
    pairs = list(combinations(network_ids, 2))

    result_list: list[pd.DataFrame] = []
    for i, j in pairs:
        df_i = all_targeting[all_targeting["network_id"] == i][["name", "targeting"]].rename(
            columns={"targeting": "targeting_i"}
        )
        df_j = all_targeting[all_targeting["network_id"] == j][["name", "targeting"]].rename(
            columns={"targeting": "targeting_j"}
        )
        df_compare = df_i.merge(df_j, on="name", how="inner")
        df_compare["compared_networks"] = f"{i}_vs_{j}"
        # Preserve the public output schema; input weights are already numeric.
        df_compare["targetingNet1"] = df_compare["targeting_i"].astype(str)
        df_compare["targetingNet2"] = df_compare["targeting_j"].astype(str)
        df_compare["deltaTargeting"] = (df_compare["targeting_i"] - df_compare["targeting_j"]).abs()
        with np.errstate(divide="ignore", invalid="ignore"):
            df_compare["log2TargetingFC"] = np.log2(df_compare["targeting_i"] / df_compare["targeting_j"])
        result_list.append(
            df_compare[
                [
                    "name",
                    "compared_networks",
                    "targetingNet1",
                    "targetingNet2",
                    "deltaTargeting",
                    "log2TargetingFC",
                ]
            ]
        )

    if not result_list:
        return pd.DataFrame(
            columns=[
                "name",
                "compared_networks",
                "targetingNet1",
                "targetingNet2",
                "deltaTargeting",
                "log2TargetingFC",
            ]
        )
    return pd.concat(result_list, ignore_index=True)


def package_data_rename(
    data: pd.DataFrame,
    source: str,
    target: str,
    condition: str,
    weight: Optional[str] = None,
) -> pd.DataFrame:
    cols = [source, target, condition] + ([weight] if weight else [])
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in input data: {missing}")

    out = data.copy()
    rename_map = {source: "source", target: "target", condition: "condition"}
    if weight:
        rename_map[weight] = "weight"

    out = out.rename(columns=rename_map)
    if "weight" not in out.columns:
        out["weight"] = 1.0

    return out[["source", "target", "condition", "weight"]]


def package_data_remap(
    data: pd.DataFrame,
    remap: Union[Dict[str, str], pd.DataFrame],
    columns: Sequence[str] = ("source", "target"),
) -> pd.DataFrame:
    out = data.copy()

    if isinstance(remap, pd.DataFrame):
        if remap.shape[1] < 2:
            raise ValueError("remap DataFrame must have at least two columns: old,new")
        mapping = dict(zip(remap.iloc[:, 0].astype(str), remap.iloc[:, 1].astype(str)))
    else:
        mapping = {str(k): str(v) for k, v in remap.items()}

    for col in columns:
        if col not in out.columns:
            raise ValueError(f"Column '{col}' not found in data")
        out[col] = out[col].astype(str).map(lambda x: mapping.get(x, x))

    return out


def package_data(
    data: pd.DataFrame,
    source: str = "source",
    target: str = "target",
    condition: str = "condition",
    weight: Optional[str] = None,
    directed: bool = True,
    drop_self_loops: bool = True,
    aggregator: str = "sum",
) -> pd.DataFrame:
    out = package_data_rename(data, source, target, condition, weight)
    out[["source", "target", "condition"]] = out[["source", "target", "condition"]].astype(str)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)

    if drop_self_loops:
        out = out[out["source"] != out["target"]].copy()

    if not directed:
        out[["source", "target"]] = out.apply(
            lambda r: pd.Series(sorted((r["source"], r["target"]))), axis=1
        )

    if aggregator not in {"sum", "mean", "max", "min"}:
        raise ValueError("aggregator must be one of: sum, mean, max, min")

    out = (
        out.groupby(["condition", "source", "target"], as_index=False)["weight"]
        .agg(aggregator)
        .reset_index(drop=True)
    )
    return out


def _edge_set(df: pd.DataFrame) -> set:
    return set(zip(df["source"], df["target"]))


def _degree_table(df: pd.DataFrame) -> pd.DataFrame:
    src = df.groupby("source", as_index=False).agg(out_degree=("target", "size"), out_weight=("weight", "sum"))
    src = src.rename(columns={"source": "node"})
    tgt = df.groupby("target", as_index=False).agg(in_degree=("source", "size"), in_weight=("weight", "sum"))
    tgt = tgt.rename(columns={"target": "node"})
    out = src.merge(tgt, on="node", how="outer").fillna(0)
    out["degree"] = out["out_degree"] + out["in_degree"]
    out["weight_degree"] = out["out_weight"] + out["in_weight"]
    return out


def dynet_internal(
    data: pd.DataFrame,
    condition_a: str,
    condition_b: str,
) -> Dict[str, Union[pd.DataFrame, dict, str]]:
    df_a = data[data["condition"] == condition_a][["source", "target", "weight"]].copy()
    df_b = data[data["condition"] == condition_b][["source", "target", "weight"]].copy()

    edges_a = _edge_set(df_a)
    edges_b = _edge_set(df_b)

    kept = edges_a & edges_b
    lost = edges_a - edges_b
    gained = edges_b - edges_a

    all_edges = []
    for s, t in kept:
        wa = float(df_a[(df_a["source"] == s) & (df_a["target"] == t)]["weight"].sum())
        wb = float(df_b[(df_b["source"] == s) & (df_b["target"] == t)]["weight"].sum())
        all_edges.append((s, t, "kept", wa, wb, wb - wa))
    for s, t in lost:
        wa = float(df_a[(df_a["source"] == s) & (df_a["target"] == t)]["weight"].sum())
        all_edges.append((s, t, "lost", wa, 0.0, -wa))
    for s, t in gained:
        wb = float(df_b[(df_b["source"] == s) & (df_b["target"] == t)]["weight"].sum())
        all_edges.append((s, t, "gained", 0.0, wb, wb))

    edge_changes = pd.DataFrame(
        all_edges,
        columns=["source", "target", "status", f"weight_{condition_a}", f"weight_{condition_b}", "delta_weight"],
    )

    deg_a = _degree_table(df_a).rename(
        columns={
            "in_degree": f"in_degree_{condition_a}",
            "out_degree": f"out_degree_{condition_a}",
            "degree": f"degree_{condition_a}",
            "weight_degree": f"weight_degree_{condition_a}",
        }
    )
    deg_b = _degree_table(df_b).rename(
        columns={
            "in_degree": f"in_degree_{condition_b}",
            "out_degree": f"out_degree_{condition_b}",
            "degree": f"degree_{condition_b}",
            "weight_degree": f"weight_degree_{condition_b}",
        }
    )

    node_changes = deg_a.merge(deg_b, on="node", how="outer").fillna(0)
    node_changes["delta_degree"] = node_changes[f"degree_{condition_b}"] - node_changes[f"degree_{condition_a}"]
    node_changes["delta_weight_degree"] = (
        node_changes[f"weight_degree_{condition_b}"] - node_changes[f"weight_degree_{condition_a}"]
    )
    node_changes["rewiring_score"] = node_changes["delta_degree"].abs() + node_changes["delta_weight_degree"].abs()
    node_changes = node_changes.sort_values("rewiring_score", ascending=False).reset_index(drop=True)

    summary = {
        "condition_a": condition_a,
        "condition_b": condition_b,
        "n_edges_a": int(len(edges_a)),
        "n_edges_b": int(len(edges_b)),
        "n_kept": int(len(kept)),
        "n_lost": int(len(lost)),
        "n_gained": int(len(gained)),
        "n_nodes": int(node_changes.shape[0]),
    }

    return {
        "comparison": f"{condition_a}_vs_{condition_b}",
        "edge_changes": edge_changes,
        "node_changes": node_changes,
        "summary": summary,
    }


def dynet_main(
    data: pd.DataFrame,
    conditions: Optional[Iterable[str]] = None,
    pairwise: str = "adjacent",
) -> Dict[str, Union[list, dict]]:
    if conditions is None:
        conditions = list(data["condition"].dropna().astype(str).unique())
    else:
        conditions = [str(c) for c in conditions]

    if len(conditions) < 2:
        raise ValueError("At least two conditions are required")

    if pairwise not in {"adjacent", "all"}:
        raise ValueError("pairwise must be 'adjacent' or 'all'")

    pairs: Sequence[Tuple[str, str]]
    if pairwise == "adjacent":
        pairs = list(zip(conditions[:-1], conditions[1:]))
    else:
        pairs = list(combinations(conditions, 2))

    comparisons = [dynet_internal(data, a, b) for a, b in pairs]

    edge_counts = pd.DataFrame(
        [
            {
                "comparison": c["comparison"],
                **{k: v for k, v in c["summary"].items() if k.startswith("n_")},
            }
            for c in comparisons
        ]
    )

    return {"comparisons": comparisons, "edge_counts": edge_counts}


def dynet_plot(
    result: Dict[str, Union[list, dict]],
    what: str = "edges",
    comparison: int = 0,
    top_n: int = 20,
    ax=None,
):
    comparisons = result["comparisons"]
    if comparison >= len(comparisons):
        raise IndexError("comparison index out of range")

    comp = comparisons[comparison]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    if what == "edges":
        counts = comp["edge_changes"]["status"].value_counts().reindex(["gained", "lost", "kept"]).fillna(0)
        colors = ["#2a9d8f", "#e76f51", "#264653"]
        ax.bar(counts.index, counts.values, color=colors)
        ax.set_title(f"Edge Rewiring: {comp['comparison']}")
        ax.set_ylabel("Edge count")
        ax.set_xlabel("Status")
    elif what == "nodes":
        top = comp["node_changes"].head(top_n).iloc[::-1]
        ax.barh(top["node"], top["rewiring_score"], color="#457b9d")
        ax.set_title(f"Top Rewired Nodes: {comp['comparison']}")
        ax.set_xlabel("Rewiring score")
        ax.set_ylabel("Node")
    else:
        raise ValueError("what must be 'edges' or 'nodes'")

    plt.tight_layout()
    return ax.figure
