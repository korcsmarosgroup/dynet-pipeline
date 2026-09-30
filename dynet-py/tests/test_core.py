import pandas as pd
import pytest

from dynet_py import (
    calculate_jaccard_indices,
    compare_targeting,
    rewiring_analysis,
    rewiring_plot,
    dynet_main,
    package_data,
    small_multiples_plot,
)


def test_package_data_standardizes_and_aggregates():
    raw = pd.DataFrame(
        {
            "src": ["A", "A", "A"],
            "dst": ["B", "B", "C"],
            "cond": ["X", "X", "X"],
            "w": [1, 2, 3],
        }
    )
    out = package_data(raw, source="src", target="dst", condition="cond", weight="w")
    assert set(out.columns) == {"condition", "source", "target", "weight"}
    assert out.shape[0] == 2
    ab = out[(out["source"] == "A") & (out["target"] == "B")]["weight"].iloc[0]
    assert ab == 3


def test_dynet_main_detects_gained_and_lost_edges():
    raw = pd.DataFrame(
        {
            "source": ["A", "A", "B", "C"],
            "target": ["B", "C", "C", "D"],
            "condition": ["T0", "T0", "T1", "T1"],
            "weight": [1, 1, 1, 1],
        }
    )
    out = dynet_main(package_data(raw), conditions=["T0", "T1"], output="edges")
    statuses = out["status"].value_counts().to_dict()
    assert statuses.get("lost", 0) == 2
    assert statuses.get("gained", 0) == 2


def test_dynet_main_all_pairwise():
    raw = pd.DataFrame(
        {
            "source": ["A", "A", "A"],
            "target": ["B", "C", "D"],
            "condition": ["C1", "C2", "C3"],
            "weight": [1, 1, 1],
        }
    )
    out = dynet_main(package_data(raw), pairwise="all", output="summary")
    assert len(out) == 3


def test_rewiring_analysis_returns_expected_columns():
    el1 = pd.DataFrame({"from": ["A", "B"], "to": ["B", "C"], "weight": [1.0, 2.0]})
    el2 = pd.DataFrame({"from": ["A", "C"], "to": ["C", "B"], "weight": [2.0, 1.0]})

    out = rewiring_analysis({"net1": el1, "net2": el2})
    assert list(out.columns) == ["name", "rewiring", "degree", "degree_corrected_rewiring"]
    assert set(out["name"]) == {"A", "B", "C"}


def test_calculate_jaccard_indices_named_networks():
    el1 = pd.DataFrame({"from": ["A"], "to": ["B"], "weight": [1.0]})
    el2 = pd.DataFrame({"from": ["A"], "to": ["C"], "weight": [1.0]})
    jac = calculate_jaccard_indices({"n1": el1, "n2": el2})
    assert jac.shape == (2, 2)
    assert list(jac.index) == ["n1", "n2"]
    assert jac.loc["n1", "n1"] == 1.0


def test_compare_targeting_columns():
    el1 = pd.DataFrame({"from": ["A", "B"], "to": ["B", "C"], "weight": [1.0, 2.0]})
    el2 = pd.DataFrame({"from": ["A", "C"], "to": ["C", "B"], "weight": [2.0, 1.0]})
    out = compare_targeting([el1, el2])
    assert list(out.columns) == [
        "name",
        "compared_networks",
        "targetingNet1",
        "targetingNet2",
        "deltaTargeting",
        "log2TargetingFC",
    ]


def test_plot_functions_return_figures():
    el1 = pd.DataFrame({"from": ["A", "B"], "to": ["B", "C"], "weight": [1.0, 2.0]})
    el2 = pd.DataFrame({"from": ["A", "C"], "to": ["C", "B"], "weight": [2.0, 1.0]})
    inputs = {"net1": el1, "net2": el2}
    rewiring = rewiring_analysis(inputs)
    fig1 = rewiring_plot(inputs, rewiring)
    fig2 = small_multiples_plot(inputs, "B")
    assert fig1 is not None
    assert fig2 is not None


def test_jaccard_compares_labels_and_ignores_matrix_order():
    a = pd.DataFrame({"from": ["A"], "to": ["B"]})
    b = pd.DataFrame({"from": ["A"], "to": ["C"]})
    jac = calculate_jaccard_indices({"ab": a, "ac": b})
    assert jac.loc["ab", "ac"] == 0.0

    matrix = pd.DataFrame([[0., 2., 0.], [0., 0., 3.], [0., 0., 0.]],
                          index=list("ABC"), columns=list("ABC"))
    reordered = matrix.loc[list("CAB"), list("CAB")]
    jac = calculate_jaccard_indices([matrix, reordered])
    assert jac.iloc[0, 1] == 1.0


def test_edge_operations_do_not_build_dense_matrices(monkeypatch):
    import matplotlib.pyplot as plt
    from dynet_py import PreparedNetworks, prepare_networks
    from dynet_py.core import _PreparedNetwork

    def unexpected_matrix(self):
        raise AssertionError("Edge operations must not allocate adjacency matrices")

    monkeypatch.setattr(_PreparedNetwork, "adjacency", unexpected_matrix)
    monkeypatch.setattr(PreparedNetworks, "adjacency_tensor", unexpected_matrix)
    # Duplicate cancellation must remove the edge but retain its nodes.
    el = pd.DataFrame({"from": ["A", "A", "B", "C"],
                       "to": ["B", "B", "C", "C"], "weight": [2., -2., -3., 4.]})
    prepared = prepare_networks({"first": el, "second": el})
    assert calculate_jaccard_indices(prepared).iloc[0, 1] == 1.0
    targeting = compare_targeting(prepared).set_index("name")
    assert targeting.loc["A", "targetingNet1"] == "0.0"
    assert targeting.loc["B", "targetingNet1"] == "0.0"
    assert targeting.loc["C", "targetingNet1"] == "1.0"
    fig = small_multiples_plot(prepared, "C")
    assert [ax.get_title() for ax in fig.axes] == ["first", "second"]
    plt.close(fig)
    scores = pd.DataFrame({"name": list("ABC"), "rewiring": [0., 1., 2.], "degree": [0., 1., 1.]})
    fig = rewiring_plot(prepared, scores)
    assert len(fig.axes[0].lines) == 1  # Positive self-loop only.
    plt.close(fig)
    fig = rewiring_plot(prepared, scores, structure_only=True)
    assert len(fig.axes[0].lines) == 2  # Negative edge participates in structure.
    plt.close(fig)


def test_prepared_inputs_are_reusable_snapshots_with_isolated_nodes():
    import matplotlib.pyplot as plt
    from dynet_py import format_indata, prepare_networks

    matrix = pd.DataFrame([[0., 2., 0.], [0., 0., 0.], [0., 0., 0.]],
                          index=list("ABC"), columns=list("ABC"))
    edge_list = pd.DataFrame({"from": ["A", "A", "C"], "to": ["B", "B", "C"],
                              "weight": [1., 1., 0.]})
    prepared = prepare_networks({"matrix": matrix, "edges": edge_list})
    baseline = rewiring_analysis(prepared)
    assert baseline["rewiring"].eq(0).all()
    assert baseline.set_index("name")["degree"].to_dict() == {"A": 1, "B": 1, "C": 0}
    matrix.iloc[0, 1] = 99.
    edge_list.loc[0, "weight"] = 99.
    exported = format_indata(prepared)
    pd.testing.assert_frame_equal(exported[0], exported[1].loc[list("ABC"), list("ABC")])
    exported[0].iloc[0, 1] = 123.
    for structural in (True, False):
        rewiring_analysis(prepared, structure_only=structural)
        fig = rewiring_plot(prepared, baseline, structure_only=structural)
        plt.close(fig)
    pd.testing.assert_frame_equal(rewiring_analysis(prepared), baseline)
    assert "C" in compare_targeting(prepared)["name"].tolist()
    assert format_indata(prepared)[0].loc["A", "B"] == 2.


def test_jaccard_duplicate_cancellation_and_empty_networks():
    cancelled = pd.DataFrame({"from": ["A", "A"], "to": ["B", "B"], "weight": [1., -1.]})
    empty = pd.DataFrame(columns=["from", "to"])
    present = pd.DataFrame({"from": ["A"], "to": ["B"], "weight": [-1.]})
    jac = calculate_jaccard_indices([cancelled, empty, present])
    assert jac.iloc[0, 1] == 1.
    assert jac.iloc[0, 2] == 0.


def test_prepared_numpy_input_is_a_snapshot():
    import numpy as np
    from dynet_py import format_indata, prepare_networks

    original = np.array([[0., 2.], [0., 0.]])
    prepared = prepare_networks([original])
    original[0, 1] = 99.
    assert format_indata(prepared)[0].iloc[0, 1] == 2.


@pytest.mark.parametrize("named", [True, False])
def test_prepared_collection_skips_parsing_across_all_consumers(monkeypatch, named):
    import matplotlib.pyplot as plt
    from dynet_py import PreparedNetworks, core, format_indata, prepare_networks

    matrix = pd.DataFrame([[0., 2., 0.], [0., 0., 3.], [0., 0., 0.]],
                          index=list("ABC"), columns=list("ABC"))
    edges = pd.DataFrame({"from": ["A", "C"], "to": ["B", "B"], "weight": [4., 1.]})
    inputs = {"matrix": matrix, "edges": edges} if named else [matrix, edges]
    expected_rewiring = {mode: rewiring_analysis(inputs, structure_only=mode) for mode in (False, True)}
    expected_targeting = compare_targeting(inputs)
    expected_jaccard = calculate_jaccard_indices(inputs)
    expected_matrices = format_indata(inputs)
    prepared = prepare_networks(inputs)
    assert isinstance(prepared, PreparedNetworks)
    assert list(prepared) == (["matrix", "edges"] if named else ["1", "2"])

    def unexpected_parse(*args, **kwargs):
        raise AssertionError("Prepared collections must bypass input parsing")

    monkeypatch.setattr(core, "_coerce_named_inputs", unexpected_parse)
    monkeypatch.setattr(core, "_prepare_network", unexpected_parse)
    monkeypatch.setattr(core, "_coerce_to_adjacency_matrix", unexpected_parse)
    monkeypatch.setattr(core, "_normalize_labels", unexpected_parse)
    monkeypatch.setattr(core, "_normalize_weights", unexpected_parse)
    conversions = []
    original = core._indata_to_edgelist

    def record_conversion(adjacency):
        conversions.append(adjacency)
        return original(adjacency)

    monkeypatch.setattr(core, "_indata_to_edgelist", record_conversion)
    assert prepare_networks(prepared) is prepared
    cached_matrices = None
    for structural in (False, True):
        pd.testing.assert_frame_equal(compare_targeting(prepared), expected_targeting)
        pd.testing.assert_frame_equal(calculate_jaccard_indices(prepared), expected_jaccard)
        result = rewiring_analysis(prepared, structure_only=structural)
        pd.testing.assert_frame_equal(result, expected_rewiring[structural])
        fig = rewiring_plot(prepared, result, structure_only=structural)
        plt.close(fig)
        for mode in ("focus_only", "r_compat"):
            fig = small_multiples_plot(prepared, "B", mode=mode)
            assert [ax.get_title() for ax in fig.axes] == list(prepared)
            plt.close(fig)
        for actual, expected in zip(format_indata(prepared), expected_matrices):
            pd.testing.assert_frame_equal(actual, expected)
        matrices = [network.adjacency() for network in prepared.values()]
        if cached_matrices is not None:
            assert all(current is cached for current, cached in zip(matrices, cached_matrices))
        cached_matrices = matrices
    assert len(conversions) == 1  # Only the matrix input needs an edge-list conversion.


def test_prepared_collection_cannot_be_changed_to_contain_raw_inputs():
    from dynet_py import prepare_networks

    edges = pd.DataFrame({"from": ["A"], "to": ["B"]})
    prepared = prepare_networks({"first": edges})
    with pytest.raises(TypeError):
        prepared["first"] = edges
    with pytest.raises(TypeError):
        del prepared["first"]
    with pytest.raises(ValueError, match="Network names must be unique"):
        prepare_networks({1: edges, "1": edges})


@pytest.mark.parametrize("input_mode", ["--input-csv", "--edge-lists"])
@pytest.mark.parametrize("include_weights", [True, False])
@pytest.mark.parametrize("endpoints", [("source", "target"), ("from", "to"), ("src", "dst")])
def test_cli_prepares_each_edge_list_once(tmp_path, monkeypatch, input_mode, include_weights, endpoints):
    import sys
    import matplotlib.pyplot as plt
    from dynet_py import cli, core

    data = pd.DataFrame({"network": ["first", "second"], "from": ["A", "A"],
                         "to": ["B", "B"], "weight": [1., 2.]})
    data = data.rename(columns={"from": endpoints[0], "to": endpoints[1]})
    if not include_weights:
        data = data.drop(columns="weight")
    sources = []
    if input_mode == "--input-csv":
        source = tmp_path / "networks.csv"
        data.to_csv(source, index=False)
        sources.append(str(source))
    else:
        for name, frame in data.groupby("network", sort=False):
            source = tmp_path / f"{name}.csv"
            frame.drop(columns="network").to_csv(source, index=False)
            sources.append(str(source))
    destination = tmp_path / "results"
    calls = []
    original = core._prepare_edgelist
    input_calls = []
    original_inputs = core._coerce_named_inputs

    def prepare_once(frame):
        calls.append(frame)
        return original(frame)

    def parse_inputs_once(inputs):
        input_calls.append(inputs)
        return original_inputs(inputs)

    monkeypatch.setattr(core, "_prepare_edgelist", prepare_once)
    monkeypatch.setattr(core, "_coerce_named_inputs", parse_inputs_once)
    monkeypatch.setattr(sys, "argv", ["dynet-py", input_mode, *sources,
                                     "--out-dir", str(destination)])
    cli.main()
    assert len(calls) == 2
    assert len(input_calls) == 1
    assert {p.name for p in destination.iterdir()} == {
        "dynet_py_output.csv", "compare_targeting.csv", "dynet_py_plot.png", "small_multiples_plot.png",
    }
    output = pd.read_csv(destination / "dynet_py_output.csv")
    assert set(output["name"]) == {"A", "B"}
    plt.close("all")
