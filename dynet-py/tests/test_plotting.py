"""Check plot selection and readability without relying on image snapshots."""

import builtins

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_rgba
from matplotlib.text import Annotation
import numpy as np
import pandas as pd
import pytest

from dynet_py import PreparedNetworks, prepare_networks, rewiring_plot, small_multiples_plot
from dynet_py import core


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def edges(*pairs):
    return pd.DataFrame(pairs, columns=["source", "target", "weight"])


def scores(names, values=None, degrees=None):
    names = list(names)
    return pd.DataFrame({
        "name": names,
        "rewiring": list(range(len(names))) if values is None else values,
        "degree": np.ones(len(names)) if degrees is None else degrees,
    })


def scatter(ax):
    return next(collection for collection in ax.collections if isinstance(collection, PathCollection))


def labels(ax):
    return {text.get_text(): np.array(text.get_position()) for text in ax.texts
            if not isinstance(text, Annotation) and text.get_text()}


def attributes_by_label(ax):
    points = scatter(ax)
    offsets = np.asarray(points.get_offsets())
    return {
        name: (points.get_array()[np.argmin(np.linalg.norm(offsets - position, axis=1))],
               points.get_sizes()[np.argmin(np.linalg.norm(offsets - position, axis=1))])
        for name, position in labels(ax).items()
    }


def directed_edges(ax):
    positions = labels(ax)

    def node_at(position):
        return next(name for name, candidate in positions.items() if np.allclose(position, candidate))

    return {(node_at(arrow.xyann), node_at(arrow.xy)) for arrow in ax.texts
            if isinstance(arrow, Annotation) and arrow.arrow_patch is not None}


def test_rewiring_top_nodes_use_supplied_scores_and_induced_edges():
    inputs = [edges(("A", "B", 1), ("B", "C", 1), ("C", "D", 1))]
    supplied = scores("ABCD", [1, 8, 6, 100], [1, 8, 4, 2])
    original = supplied.copy(deep=True)
    fig = rewiring_plot(inputs, supplied, top_n=2, labels="all", layout="circular", figsize=(6, 4))
    ax = fig.axes[0]
    assert set(labels(ax)) == {"B", "D"}
    assert len(ax.lines) == 0  # A ranking filter must not create a B--D edge.
    actual = attributes_by_label(ax)
    assert actual["B"][0] == 8
    assert actual["D"][0] == 100
    assert actual["B"][1] > actual["D"][1]  # Full supplied degree is retained.
    np.testing.assert_allclose(fig.get_size_inches(), [6, 4])
    pd.testing.assert_frame_equal(supplied, original)


def test_rewiring_focus_uses_union_hops_before_ranking_and_retains_focus():
    inputs = {
        "first": edges(("A", "B", 1), ("C", "D", 1), ("X", "Y", 1)),
        "second": edges(("B", "C", 1)),
    }
    supplied = scores("ABCDXY", [0, 5, 8, 100, 200, 0])
    fig = rewiring_plot(inputs, supplied, focus_node="A", radius=2, top_n=2,
                        labels="all", layout="circular")
    assert set(labels(fig.axes[0])) == {"A", "C"}
    assert len(fig.axes[0].lines) == 0
    fig = rewiring_plot(inputs, supplied, focus_node="A", radius=3, top_n=2,
                        labels="all", layout="circular")
    assert set(labels(fig.axes[0])) == {"A", "D"}
    assert attributes_by_label(fig.axes[0])["D"][0] == 100


@pytest.mark.parametrize("structural, expected", [(False, {"A"}), (True, {"A", "B", "C"})])
def test_rewiring_neighborhood_matches_edge_sign_mode(structural, expected):
    inputs = [edges(("A", "B", -1), ("B", "C", 1), ("A", "Z", 0))]
    fig = rewiring_plot(inputs, scores("ABCZ"), structure_only=structural,
                        focus_node="A", radius=2, labels="all", layout="circular")
    assert set(labels(fig.axes[0])) == expected


def test_rewiring_radius_zero_and_numeric_node_labels():
    inputs = [edges((1, 2, 1), (2, 3, 1))]
    supplied = scores([1, 2, 3], [8, 3, 2])
    fig = rewiring_plot(inputs, supplied, focus_node=2, radius=0,
                        labels=[2], layout="circular")
    assert set(labels(fig.axes[0])) == {"2"}
    assert len(scatter(fig.axes[0]).get_offsets()) == 1
    assert attributes_by_label(fig.axes[0])["2"][0] == 3


def test_rewiring_tied_ranking_is_stable_in_prepared_node_order():
    inputs = [edges(("C", "A", 1), ("A", "B", 1), ("B", "D", 1))]
    prepared = prepare_networks(inputs)
    fig = rewiring_plot(prepared, scores(prepared.node_names, [1] * 4), top_n=2,
                        labels="all", layout="circular")
    assert set(labels(fig.axes[0])) == set(prepared.node_names[:2])


def test_rewiring_auto_labels_large_graph_only_label_highest_scores():
    names = [f"n{i:02d}" for i in range(35)]
    inputs = [edges(*[(source, target, 1) for source, target in zip(names, names[1:])])]
    fig = rewiring_plot(inputs, scores(names), layout="circular")
    assert len(scatter(fig.axes[0]).get_offsets()) == 35
    assert set(labels(fig.axes[0])) == set(names[-10:])
    fig = rewiring_plot(inputs, scores(names), top_n=4, layout="circular")
    assert set(labels(fig.axes[0])) == set(names[-4:])


@pytest.mark.parametrize("option, expected", [("all", {"A", "B", "C"}), ("none", set()),
                                               (["A", "C"], {"A", "C"})])
def test_rewiring_label_selection_only_changes_labels(option, expected):
    inputs = [edges(("A", "B", 1), ("B", "C", 1))]
    fig = rewiring_plot(inputs, scores("ABC"), labels=option, layout="circular")
    assert set(labels(fig.axes[0])) == expected
    assert len(scatter(fig.axes[0]).get_offsets()) == 3
    assert len(fig.axes[0].lines) == 2


def test_rewiring_unknown_focus_has_clear_error():
    with pytest.raises(ValueError, match="focus"):
        rewiring_plot([edges(("A", "B", 1))], scores("AB"), focus_node="missing")


@pytest.mark.parametrize("kwargs", [
    {"layout": "unknown"}, {"labels": "everything"}, {"top_n": 0}, {"top_n": -1},
    {"top_n": 1.5}, {"top_n": True}, {"radius": -1}, {"radius": 1.5}, {"radius": True},
])
def test_rewiring_rejects_invalid_display_options(kwargs):
    with pytest.raises((TypeError, ValueError)):
        rewiring_plot([edges(("A", "B", 1))], scores("AB"), **kwargs)


def test_small_multiples_union_neighborhood_has_shared_positions_and_panel_edges():
    inputs = {
        "first": edges(("A", "B", 1), ("C", "D", 1)),
        "second": edges(("B", "C", -1), ("D", "E", 1)),
    }
    fig = small_multiples_plot(inputs, "A", radius=2, labels="all", layout="circular")
    first, second = fig.axes
    assert [ax.get_title() for ax in fig.axes] == ["first", "second"]
    assert set(labels(first)) == set(labels(second)) == {"A", "B", "C"}
    for node in labels(first):
        np.testing.assert_allclose(labels(first)[node], labels(second)[node])
    assert directed_edges(first) == {("A", "B")}
    assert directed_edges(second) == {("B", "C")}


def test_small_multiples_neighborhood_contains_edges_between_neighbors():
    inputs = [edges(("A", "B", 1), ("C", "A", 1), ("B", "C", 1), ("C", "D", 1))]
    fig = small_multiples_plot(inputs, "A", labels="all", layout="circular")
    assert set(labels(fig.axes[0])) == {"A", "B", "C"}
    assert directed_edges(fig.axes[0]) == {("A", "B"), ("C", "A"), ("B", "C")}


def test_small_multiples_cap_uses_union_degree_and_retains_focus():
    inputs = {
        "first": edges(("A", "B", 1), ("B", "C", 1)),
        "second": edges(("C", "D", 1), ("C", "E", 1), ("C", "F", 1), ("C", "G", 1)),
    }
    fig = small_multiples_plot(inputs, "A", radius=3, max_nodes=2,
                               labels="all", layout="circular")
    assert all(set(labels(ax)) == {"A", "C"} for ax in fig.axes)
    assert directed_edges(fig.axes[0]) == set()
    assert directed_edges(fig.axes[1]) == set()


@pytest.mark.parametrize("focus", ["missing", "isolated"])
def test_small_multiples_missing_or_isolated_focus_is_still_visible(focus):
    matrix = pd.DataFrame([[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                          index=["A", "B", "isolated"], columns=["A", "B", "isolated"])
    fig = small_multiples_plot({"matrix": matrix}, focus, labels="all", layout="circular")
    assert set(labels(fig.axes[0])) == {focus}
    assert len(scatter(fig.axes[0]).get_offsets()) == 1
    assert directed_edges(fig.axes[0]) == set()


def test_small_multiples_auto_labels_large_neighborhood_only_label_focus():
    inputs = [edges(*[("A", f"n{i:02d}", 1) for i in range(35)])]
    fig = small_multiples_plot(inputs, "A", layout="circular")
    assert set(labels(fig.axes[0])) == {"A"}
    assert len(scatter(fig.axes[0]).get_offsets()) == 36


def test_small_multiples_radius_zero_numeric_labels_and_custom_size():
    inputs = [edges((1, 2, 1), (2, 3, 1))]
    fig = small_multiples_plot(inputs, 2, radius=0, labels=[2],
                               layout="circular", figsize=(7, 3))
    assert set(labels(fig.axes[0])) == {"2"}
    assert directed_edges(fig.axes[0]) == set()
    np.testing.assert_allclose(fig.get_size_inches(), [7, 3])


def test_small_multiples_r_compat_keeps_distant_and_negative_edges():
    inputs = [edges(("A", "B", 1), ("B", "C", -1), ("X", "Y", 2), ("Y", "Z", 0))]
    fig = small_multiples_plot(inputs, "A", mode="r_compat", labels="all", layout="circular")
    assert directed_edges(fig.axes[0]) == {("A", "B"), ("B", "C"), ("X", "Y")}


@pytest.mark.parametrize("kwargs", [
    {"layout": "unknown"}, {"labels": "everything"}, {"mode": "unknown"},
    {"max_nodes": 0}, {"max_nodes": -1}, {"max_nodes": 1.5}, {"max_nodes": True},
    {"radius": -1}, {"radius": 1.5}, {"radius": True},
    {"mode": "r_compat", "radius": 2},
])
def test_small_multiples_rejects_invalid_display_options(kwargs):
    with pytest.raises((TypeError, ValueError)):
        small_multiples_plot([edges(("A", "B", 1))], "A", **kwargs)


@pytest.mark.parametrize("inputs", [[], [edges()]])
def test_empty_network_unions_are_safe(inputs):
    fig = rewiring_plot(inputs, scores([]), layout="circular")
    assert len(scatter(fig.axes[0]).get_offsets()) == 0
    assert labels(fig.axes[0]) == {}
    fig = small_multiples_plot(inputs, "focus", layout="circular")
    assert fig is not None


def test_zero_edge_union_retains_isolated_nodes_and_finite_marker_sizes():
    inputs = [edges(("A", "B", 1), ("A", "B", -1), ("C", "C", 0))]
    fig = rewiring_plot(inputs, scores("ABC", [0, 0, 0], [0, 0, 0]),
                        labels="all", layout="circular")
    assert set(labels(fig.axes[0])) == set("ABC")
    assert len(fig.axes[0].lines) == 0
    assert np.isfinite(scatter(fig.axes[0]).get_sizes()).all()
    assert (scatter(fig.axes[0]).get_sizes() > 0).all()


def test_plotting_prepared_networks_never_reparses_or_allocates_dense_adjacency(monkeypatch):
    prepared = prepare_networks({
        "first": edges(("A", "B", 1), ("B", "C", 1)),
        "second": edges(("C", "D", 1)),
    })

    def unexpected(*args, **kwargs):
        raise AssertionError("Plotting prepared edges must not parse inputs or allocate dense matrices")

    monkeypatch.setattr(core, "_prepare_network", unexpected)
    monkeypatch.setattr(core, "_coerce_named_inputs", unexpected)
    monkeypatch.setattr(core._PreparedNetwork, "adjacency", unexpected)
    monkeypatch.setattr(PreparedNetworks, "adjacency_tensor", unexpected)
    rewiring_plot(prepared, scores("ABCD"), focus_node="A", radius=2, top_n=2, layout="circular")
    small_multiples_plot(prepared, "A", radius=2, max_nodes=2, layout="circular")
    assert prepared._adjacency_tensor is None
    assert all(network._adjacency is None for network in prepared.values())


def test_auto_layout_falls_back_when_optional_plot_dependency_is_unavailable(monkeypatch):
    original_import = builtins.__import__

    def without_networkx(name, *args, **kwargs):
        if name == "networkx" or name.startswith("networkx."):
            raise ImportError("NetworkX is deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_networkx)
    names = [f"n{i:02d}" for i in range(35)]
    inputs = [edges(*[(source, target, 1) for source, target in zip(names, names[1:])])]
    automatic = rewiring_plot(inputs, scores(names), layout="auto")
    circular = rewiring_plot(inputs, scores(names), layout="circular")
    np.testing.assert_allclose(scatter(automatic.axes[0]).get_offsets(),
                               scatter(circular.axes[0]).get_offsets())
    with pytest.raises(ImportError, match="NetworkX|networkx|plot"):
        rewiring_plot(inputs, scores(names), layout="spring")


@pytest.mark.parametrize("layout", ["spring", "kamada_kawai"])
def test_optional_layouts_are_repeatable_and_shared_between_panels(layout):
    pytest.importorskip("networkx")
    pytest.importorskip("scipy")
    inputs = {
        "first": edges(("A", "B", 1), ("B", "C", 1)),
        "second": edges(("C", "D", 1), ("D", "A", 1)),
    }
    first = rewiring_plot(inputs, scores("ABCD"), layout=layout, seed=7, labels="all")
    second = rewiring_plot(inputs, scores("ABCD"), layout=layout, seed=7, labels="all")
    np.testing.assert_allclose(scatter(first.axes[0]).get_offsets(), scatter(second.axes[0]).get_offsets())
    panels = small_multiples_plot(inputs, "A", radius=2, layout=layout, seed=7, labels="all")
    assert set(labels(panels.axes[0])) == set(labels(panels.axes[1])) == set("ABCD")
    for node in "ABCD":
        np.testing.assert_allclose(labels(panels.axes[0])[node], labels(panels.axes[1])[node])


def test_spring_layout_seed_controls_positions():
    pytest.importorskip("networkx")
    pytest.importorskip("scipy")
    inputs = [edges(("A", "B", 1), ("B", "C", 1), ("C", "D", 1))]
    first = rewiring_plot(inputs, scores("ABCD"), layout="spring", seed=3)
    second = rewiring_plot(inputs, scores("ABCD"), layout="spring", seed=4)
    assert not np.allclose(scatter(first.axes[0]).get_offsets(), scatter(second.axes[0]).get_offsets())


@pytest.mark.parametrize("values", [[np.nan, np.inf, -np.inf, 4], [np.nan, np.inf, -np.inf, np.nan]])
def test_nonfinite_rewiring_scores_remain_visible_as_gray_nodes(values):
    inputs = [edges(("A", "B", 1), ("B", "C", 1), ("C", "D", 1))]
    fig = rewiring_plot(inputs, scores("ABCD", values), labels="all", layout="circular")
    fig.canvas.draw()
    points = scatter(fig.axes[0])
    assert set(labels(fig.axes[0])) == set("ABCD")
    assert len(points.get_offsets()) == 4
    assert not np.ma.getmaskarray(points.get_offsets()).any()
    for name, value in zip("ABCD", values):
        position = labels(fig.axes[0])[name]
        index = np.argmin(np.linalg.norm(np.asarray(points.get_offsets()) - position, axis=1))
        if not np.isfinite(value):
            np.testing.assert_allclose(points.get_facecolors()[index], to_rgba("#bdbdbd"))
    assert np.isfinite(points.get_sizes()).all()


def test_nan_scores_rank_after_non_nan_scores_for_top_node_selection():
    inputs = [edges(("A", "B", 1), ("B", "C", 1), ("C", "D", 1))]
    fig = rewiring_plot(inputs, scores("ABCD", [np.nan, np.inf, -np.inf, 4]),
                        top_n=3, labels="all", layout="circular")
    assert set(labels(fig.axes[0])) == set("BCD")


@pytest.mark.parametrize("selection, expected", [({"focus_node": "B"}, set("ABC")),
                                                 ({"top_n": 2}, set("CD"))])
def test_filtered_rewiring_plot_keeps_full_graph_color_scale(selection, expected):
    inputs = [edges(("A", "B", 1), ("B", "C", 1), ("C", "D", 1))]
    supplied = scores("ABCD", [0, 1, 2, 100])
    full = rewiring_plot(inputs, supplied, labels="all", layout="circular")
    filtered = rewiring_plot(inputs, supplied, labels="all", layout="circular", **selection)
    assert set(labels(filtered.axes[0])) == expected
    assert scatter(full.axes[0]).get_clim() == scatter(filtered.axes[0]).get_clim() == (0, 100)
    np.testing.assert_allclose(scatter(full.axes[0]).to_rgba([1]), scatter(filtered.axes[0]).to_rgba([1]))


def test_large_edge_collection_preserves_filtered_induced_graph():
    names = [f"n{i:02d}" for i in range(25)]
    # Reverse duplicates exercise undirected union de-duplication as well.
    inputs = [edges(*[(source, target, 1) for source in names for target in names if source != target])]
    fig = rewiring_plot(inputs, scores(names), top_n=22, labels="all", layout="circular")
    ax = fig.axes[0]
    selected = set(names[-22:])
    assert set(labels(ax)) == selected
    collections = [collection for collection in ax.collections if isinstance(collection, LineCollection)]
    assert len(collections) == 1
    positions = labels(ax)

    def node_at(position):
        return next(name for name, candidate in positions.items() if np.allclose(position, candidate))

    actual = {frozenset(node_at(point) for point in segment) for segment in collections[0].get_segments()}
    expected = {frozenset((source, target)) for source in selected for target in selected if source != target}
    assert len(collections[0].get_segments()) == 231
    assert actual == expected


def test_rewiring_auto_labels_keep_low_scoring_focus_within_ten_label_cap():
    names = ["focus"] + [f"n{i:02d}" for i in range(35)]
    inputs = [edges(*[("focus", node, 1) for node in names[1:]])]
    fig = rewiring_plot(inputs, scores(names), focus_node="focus", layout="circular")
    assert len(scatter(fig.axes[0]).get_offsets()) == 36
    assert set(labels(fig.axes[0])) == {"focus", *names[-9:]}


@pytest.mark.parametrize("supplied", [
    pd.DataFrame({"node": ["A", "B"], "rewiring": [1, 2], "degree": [1, 1]}),
    scores([1, "1"]),
    scores([None, "B"]),
    scores("AB", ["invalid", 1]),
    scores("AB", degrees=[1j, 2]),
    pd.DataFrame([["A", 1, 1, 1]], columns=["name", "rewiring", "degree", "degree"]),
    {"name": ["A"], "rewiring": [1], "degree": [1]},
])
def test_rewiring_rejects_ambiguous_or_invalid_score_tables(supplied):
    with pytest.raises(ValueError):
        rewiring_plot([edges(("A", "B", 1))], supplied, layout="circular")
