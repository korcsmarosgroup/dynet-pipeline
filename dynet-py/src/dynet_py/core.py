from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

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
        names = [str(k) for k in input_list.keys()]
        values = list(input_list.values())
        return values, names

    if isinstance(input_list, (str, bytes)):
        raise ValueError("Input must be a sequence/mapping of matrices, edge lists, or graphs.")

    values = list(input_list)
    names = [str(i + 1) for i in range(len(values))]
    return values, names


def _set_rownames_to_colnames(mat: pd.DataFrame) -> pd.DataFrame:
    out = mat.copy()
    out.columns = [str(c) for c in out.columns]
    out.index = out.columns
    return out


def _edgelist_to_adjacency(el: pd.DataFrame) -> pd.DataFrame:
    if not {"from", "to"}.issubset(el.columns):
        raise ValueError("Edge list input must contain 'from' and 'to' columns.")

    if "weight" not in el.columns:
        work = el.copy()
        work["weight"] = 1.0
    else:
        work = el.copy()

    work["from"] = work["from"].astype(str)
    work["to"] = work["to"].astype(str)
    work["weight"] = pd.to_numeric(work["weight"], errors="coerce").fillna(0.0)

    nodes = _ordered_unique(list(work["from"]) + list(work["to"]))
    adj = pd.DataFrame(0.0, index=nodes, columns=nodes)
    grouped = work.groupby(["from", "to"], as_index=False)["weight"].sum()
    for _, row in grouped.iterrows():
        adj.loc[row["from"], row["to"]] = float(row["weight"])
    return adj


def _is_graph_like(item: object) -> bool:
    return hasattr(item, "nodes") and hasattr(item, "edges")


def _graph_to_adjacency(graph: object) -> pd.DataFrame:
    node_names = [str(n) for n in list(graph.nodes())]
    if not node_names:
        return pd.DataFrame()
    adj = pd.DataFrame(0.0, index=node_names, columns=node_names)
    for u, v, data in graph.edges(data=True):
        w = float(data.get("weight", 1.0)) if isinstance(data, dict) else 1.0
        adj.loc[str(u), str(v)] = adj.loc[str(u), str(v)] + w
        if not getattr(graph, "is_directed", lambda: True)():
            adj.loc[str(v), str(u)] = adj.loc[str(v), str(u)] + w
    return adj


def _coerce_to_adjacency_matrix(item: object) -> pd.DataFrame:
    if isinstance(item, pd.DataFrame):
        if _is_edge_list_df(item):
            return _edgelist_to_adjacency(item)
        if item.shape[0] != item.shape[1]:
            raise ValueError("Adjacency matrix data frames must be square.")
        out = _set_rownames_to_colnames(item)
        return out.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if isinstance(item, (np.ndarray, list, tuple)):
        arr = np.asarray(item, dtype=float)
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
    items, _ = _coerce_named_inputs(input_list)
    return [_coerce_to_adjacency_matrix(item) for item in items]


def _expand_adjacency_matrices(adj_matrices: Sequence[pd.DataFrame]) -> list[pd.DataFrame]:
    all_nodes = _ordered_unique(node for m in adj_matrices for node in m.index.astype(str))
    expanded: list[pd.DataFrame] = []
    for m in adj_matrices:
        current = m.copy()
        current.index = current.index.astype(str)
        current.columns = current.columns.astype(str)
        base = pd.DataFrame(0.0, index=all_nodes, columns=all_nodes)
        base.loc[current.index, current.columns] = current.values
        expanded.append(base)
    return expanded


def _union_adjacency_matrices(matrices: Sequence[pd.DataFrame]) -> pd.DataFrame:
    all_nodes = _ordered_unique(node for m in matrices for node in m.index.astype(str))
    union = pd.DataFrame(0, index=all_nodes, columns=all_nodes, dtype=int)
    for adj in matrices:
        nodes = list(adj.index.astype(str))
        arr = adj.loc[nodes, nodes].to_numpy(dtype=float)
        mask = arr > 0
        if np.any(mask):
            sub = union.loc[nodes, nodes].to_numpy(dtype=int).copy()
            sub[mask] = 1
            union.loc[nodes, nodes] = sub
    return union


def _indata_to_edgelist(ingraph: pd.DataFrame) -> pd.DataFrame:
    arr = ingraph.to_numpy(dtype=float)
    rows, cols = np.where(arr != 0)
    return pd.DataFrame(
        {
            "from": ingraph.index.to_numpy()[rows],
            "to": ingraph.columns.to_numpy()[cols],
            "weight": arr[rows, cols],
        }
    )


def _sum_matrices(matrices: Sequence[pd.DataFrame]) -> pd.DataFrame:
    result = pd.DataFrame(0.0, index=matrices[0].index, columns=matrices[0].columns)
    counter = pd.DataFrame(0.0, index=matrices[0].index, columns=matrices[0].columns)
    for m in matrices:
        result = result + m
        counter = counter + (m != 0).astype(float)
    out = result / counter
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _square_matrices(matrices: Sequence[pd.DataFrame]) -> list[pd.DataFrame]:
    return [m**2 for m in matrices]


def _centroid_distance(matrices: Sequence[pd.DataFrame]) -> list[pd.Series]:
    return [m.sum(axis=1) + m.sum(axis=0) - np.diag(m.to_numpy(dtype=float)) for m in matrices]


def _structure_format(inmat: pd.DataFrame) -> pd.DataFrame:
    arr = inmat.to_numpy(dtype=float)
    out = np.where(arr != 0, 1.0, arr)
    return pd.DataFrame(out, index=inmat.index, columns=inmat.columns)


def rewiring_analysis(
    matrix_list: Union[Mapping[str, object], Sequence[object]],
    structure_only: bool = False,
) -> pd.DataFrame:
    matrix_list_fmt = format_indata(matrix_list)
    if len(matrix_list_fmt) < 2:
        raise ValueError("rewiring_analysis requires at least two input networks.")

    if structure_only:
        matrix_list_str = [_structure_format(m) for m in matrix_list_fmt]
    else:
        matrix_list_str = matrix_list_fmt

    expanded = _expand_adjacency_matrices(matrix_list_str)
    non_zero_mean = _sum_matrices(expanded)

    standardised = [m / non_zero_mean for m in expanded]
    standardised_no_nan = [m.mask(m.isna(), 0.0) for m in standardised]

    standard_sums = standardised_no_nan[0].copy()
    for m in standardised_no_nan[1:]:
        standard_sums = standard_sums + m
    centroid = standard_sums / len(standardised_no_nan)

    standard_minus_centroid = [m - centroid for m in standardised_no_nan]
    squared = _square_matrices(standard_minus_centroid)
    cent_dist = _centroid_distance(squared)
    cent_dist_bound = np.vstack([s.to_numpy(dtype=float) for s in cent_dist])
    cent_final_dist = cent_dist_bound.sum(axis=0)
    rewiring = cent_final_dist / (len(cent_dist) - 1)

    unimatrix = _union_adjacency_matrices(matrix_list_str)
    out_degree = unimatrix.sum(axis=1).to_numpy(dtype=float)
    in_degree = unimatrix.sum(axis=0).to_numpy(dtype=float)
    self_loops = np.diag(unimatrix.to_numpy(dtype=float))
    degree = out_degree + in_degree - self_loops
    degree_df = pd.DataFrame({"name": unimatrix.index.astype(str), "degree": degree})

    rewiring_df = pd.DataFrame(
        {
            "name": expanded[0].columns.astype(str),
            "rewiring": rewiring,
        }
    )
    output = rewiring_df.merge(degree_df, on="name", how="left")
    output["degree_corrected_rewiring"] = output["rewiring"] / output["degree"]
    return output


def rewiring_plot(
    matrix_list: Union[Mapping[str, object], Sequence[object]],
    output_dataframe: pd.DataFrame,
    structure_only: bool = False,
):
    matrix_list_fmt = format_indata(matrix_list)
    matrix_list_str = [_structure_format(m) for m in matrix_list_fmt] if structure_only else matrix_list_fmt
    unimatrix = _union_adjacency_matrices(matrix_list_str)

    attr = output_dataframe.set_index("name")
    node_names = list(unimatrix.index.astype(str))
    rewiring = np.array([float(attr["rewiring"].get(n, 0.0)) for n in node_names], dtype=float)
    degree = np.array([float(attr["degree"].get(n, 0.0)) for n in node_names], dtype=float)
    sizes = 300 + 1200 * (degree / degree.max() if degree.max() > 0 else degree + 1)
    n = max(1, len(node_names))
    angles = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
    pos = {node_names[i]: (np.cos(angles[i]), np.sin(angles[i])) for i in range(n)}
    edges: list[tuple[str, str]] = []
    for i, src in enumerate(node_names):
        for j, dst in enumerate(node_names):
            if j < i:
                continue
            if unimatrix.iloc[i, j] != 0 or unimatrix.iloc[j, i] != 0:
                edges.append((src, dst))

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
    items, names = _coerce_named_inputs(input_list)
    matrices = [_coerce_to_adjacency_matrix(item) for item in items]

    edges_by_network: list[pd.DataFrame] = []
    if mode not in {"focus_only", "r_compat"}:
        raise ValueError("mode must be 'focus_only' or 'r_compat'")

    for matrix, name in zip(matrices, names):
        el = _indata_to_edgelist(matrix)[["from", "to"]].drop_duplicates()
        if mode == "focus_only":
            el = el[(el["from"].astype(str) == str(focus_node)) | (el["to"].astype(str) == str(focus_node))]
        el = el.copy()
        el["id"] = name
        edges_by_network.append(el)

    all_edges = pd.concat(edges_by_network, ignore_index=True) if edges_by_network else pd.DataFrame()
    plot_ids = names

    n = max(1, len(plot_ids))
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for idx, network_id in enumerate(plot_ids):
        ax = axes_flat[idx]
        subset = all_edges[all_edges["id"] == network_id] if not all_edges.empty else pd.DataFrame()
        if subset.empty:
            nodes = [str(focus_node)]
            edges = []
        else:
            nodes = _ordered_unique(
                [str(focus_node)] + subset["from"].astype(str).tolist() + subset["to"].astype(str).tolist()
            )
            edges = list(subset[["from", "to"]].astype(str).itertuples(index=False, name=None))

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
        colors = ["#d62728" if node == str(focus_node) else "#1f77b4" for node in nodes]
        ax.scatter([pos[nm][0] for nm in nodes], [pos[nm][1] for nm in nodes], c=colors, s=400, zorder=2)
        for nm in nodes:
            ax.text(pos[nm][0], pos[nm][1], nm, fontsize=9, fontweight="bold", ha="center", va="center", zorder=3)
        ax.set_title(str(network_id))
        ax.set_axis_off()
        ax.set_aspect("equal")

    for idx in range(len(plot_ids), len(axes_flat)):
        axes_flat[idx].set_axis_off()

    fig.tight_layout()
    return fig


def _jaccard_index(adj1: pd.DataFrame, adj2: pd.DataFrame) -> float:
    a1 = adj1.to_numpy(dtype=float)
    a2 = adj2.to_numpy(dtype=float)
    set1 = set(map(tuple, np.argwhere(a1 != 0)))
    set2 = set(map(tuple, np.argwhere(a2 != 0)))
    union = set1 | set2
    intersection = set1 & set2
    if not union:
        return 1.0
    return len(intersection) / len(union)


def calculate_jaccard_indices(
    networks: Union[Mapping[str, object], Sequence[object]],
) -> pd.DataFrame:
    matrices, names = _coerce_named_inputs(networks)
    formatted = [_coerce_to_adjacency_matrix(item) for item in matrices]
    n = len(formatted)
    network_names = names if names else [f"Network_{i}" for i in range(1, n + 1)]

    jaccard_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            val = _jaccard_index(formatted[i], formatted[j])
            jaccard_matrix[i, j] = val
            jaccard_matrix[j, i] = val
    return pd.DataFrame(jaccard_matrix, index=network_names, columns=network_names)


def compare_targeting(
    input_list: Union[Mapping[str, object], Sequence[object]],
) -> pd.DataFrame:
    formatted = format_indata(input_list)

    targeting_frames: list[pd.DataFrame] = []
    for i, adj in enumerate(formatted, start=1):
        in_targeting = adj.sum(axis=0)
        frame = pd.DataFrame({"name": in_targeting.index.astype(str), "targeting": in_targeting.to_numpy(dtype=float)})
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
