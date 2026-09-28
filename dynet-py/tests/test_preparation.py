import numpy as np
import pandas as pd
import pytest

from dynet_py import (
    calculate_jaccard_indices,
    compare_targeting,
    format_indata,
    prepare_networks,
    rewiring_analysis,
)


class GraphInput:
    """The graph-like input protocol, without an optional networkx dependency."""

    def __init__(self, nodes, edges):
        self._nodes = nodes
        self._edges = edges

    def nodes(self):
        return self._nodes

    def edges(self, data=False):
        return self._edges

    def is_directed(self):
        return True


@pytest.mark.parametrize("kind", ["edges", "frame", "array", "list", "tuple", "graph"])
def test_input_formats_produce_the_same_normalized_network(kind):
    expected = pd.DataFrame([[0., 2., 0.], [0., 0., 0.], [0., 0., 0.]],
                            index=list("012"), columns=list("012"))
    matrix = [["0", "2", "0"], ["0", "0", "0"], ["0", "0", "0"]]
    inputs = {
        "edges": pd.DataFrame({"from": [0, 2], "to": [1, 2], "weight": ["2", "0"]}),
        "frame": pd.DataFrame(matrix, index=[0, 1, 2], columns=[0, 1, 2]),
        "array": np.array(matrix),
        "list": matrix,
        "tuple": tuple(tuple(row) for row in matrix),
        "graph": GraphInput([0, 1, 2], [(0, 1, {"weight": "2"})]),
    }
    prepared = prepare_networks({7: inputs[kind], "reference": expected})
    assert list(prepared) == ["7", "reference"]
    actual = format_indata(prepared)[0].loc[expected.index, expected.columns]
    pd.testing.assert_frame_equal(actual, expected)
    network = prepared["7"]
    assert all(isinstance(node, str) for node in network.nodes)
    assert network.edges()["weight"].dtype == np.dtype("float64")
    assert list(network.edges().itertuples(index=False, name=None)) == [("0", "1", 2.)]
    assert rewiring_analysis(prepared)["rewiring"].eq(0).all()
    assert compare_targeting(prepared)["deltaTargeting"].eq(0).all()
    assert calculate_jaccard_indices(prepared).loc["7", "reference"] == 1.


@pytest.mark.parametrize("bad_weight", ["invalid", None, pd.NA, np.nan, np.inf, -np.inf,
                                       1 + 2j, np.complex128(1 + 2j), np.datetime64("2026-01-01")])
@pytest.mark.parametrize("kind", ["edges", "frame", "array", "graph"])
def test_invalid_weights_are_rejected_during_preparation(kind, bad_weight):
    inputs = {
        "edges": pd.DataFrame({"from": ["A"], "to": ["B"], "weight": [bad_weight]}),
        "frame": pd.DataFrame([[0, bad_weight], [0, 0]], index=list("AB"), columns=list("AB")),
        "array": np.array([[0, bad_weight], [0, 0]], dtype=object),
        "graph": GraphInput(["A", "B"], [("A", "B", {"weight": bad_weight})]),
    }
    with pytest.raises(ValueError, match="Network 'bad': Weights must be finite real numbers"):
        prepare_networks({"bad": inputs[kind]})


@pytest.mark.parametrize("matrix", [np.ones((2, 2), dtype=complex),
                                    np.full((2, 2), np.datetime64("2026-01-01"))])
def test_non_real_matrix_dtypes_are_rejected(matrix):
    with pytest.raises(ValueError, match="Weights must be finite real numbers"):
        prepare_networks([matrix])


@pytest.mark.parametrize("network, message", [
    (pd.DataFrame({"from": [None], "to": ["B"]}), "missing values"),
    (pd.DataFrame({"from": [1], "to": ["1"]}), "unique after conversion"),
    (pd.DataFrame([["A", "B", 1, 2]], columns=["from", "to", "weight", "weight"]), "column names must be unique"),
    (pd.DataFrame([[0, 1], [0, 0]], index=list("AB"), columns=[1, "1"]), "unique after conversion"),
    (pd.DataFrame([[0, 1], [0, 0]], index=["A", "A"], columns=list("AB")), "unique after conversion"),
    (pd.DataFrame([[0, 1], [0, 0]], index=["A", None], columns=list("AB")), "missing values"),
    (pd.DataFrame([[0, 1], [0, 0]], index=list("AC"), columns=list("AB")), "must name the same nodes"),
    (pd.DataFrame([[0, 1, 2], [0, 0, 0]]), "must be square"),
    (np.zeros((2, 3)), "2D square"),
    (np.zeros(3), "2D square"),
    (GraphInput([1, "1"], []), "unique after conversion"),
    (GraphInput(["A", None], []), "missing values"),
    (GraphInput(["A"], [("A", "B", {})]), "endpoints must be present"),
])
def test_invalid_network_structure_is_rejected(network, message):
    with pytest.raises(ValueError, match=f"Network 'bad': .*{message}"):
        prepare_networks({"bad": network})


def test_missing_network_name_is_rejected():
    with pytest.raises(ValueError, match="Network names must not contain missing values"):
        prepare_networks({None: np.zeros((2, 2))})


def test_matrix_rows_are_aligned_by_labels_instead_of_relabelled():
    expected = pd.DataFrame([[0., 2., 0.], [0., 0., 3.], [0., 0., 0.]],
                            index=list("ABC"), columns=list("ABC"))
    rows_reordered = expected.loc[list("CAB")].astype("int64")
    prepared = prepare_networks({"reordered": rows_reordered, "reference": expected})
    pd.testing.assert_frame_equal(format_indata(prepared)[0], expected)
    assert calculate_jaccard_indices(prepared).iloc[0, 1] == 1.
    assert rewiring_analysis(prepared)["rewiring"].eq(0).all()
    assert list(rows_reordered.index) == list("CAB")  # Parsing owns its data.


def test_matrix_with_default_row_index_inherits_column_labels():
    matrix = pd.DataFrame([[0, 2], [0, 0]], columns=list("AB"))
    expected = pd.DataFrame([[0., 2.], [0., 0.]], index=list("AB"), columns=list("AB"))
    pd.testing.assert_frame_equal(format_indata([matrix])[0], expected)


def test_missing_weight_column_defaults_to_one_and_duplicates_are_summed():
    edges = pd.DataFrame({"from": [1, 1], "to": [2, 2]}, index=[5, 9])
    result = format_indata([edges])[0]
    assert result.loc["1", "2"] == 2.
    assert result.to_numpy().dtype == np.dtype("float64")


def test_duplicate_weight_overflow_is_rejected():
    edges = pd.DataFrame({"from": ["A", "A"], "to": ["B", "B"], "weight": [1e308, 1e308]})
    with pytest.raises(ValueError, match="Summed edge weights must be finite"):
        prepare_networks([edges])


@pytest.mark.parametrize("input_mode", ["--input-csv", "--edge-lists"])
def test_cli_rejects_invalid_weights_before_writing_outputs(tmp_path, monkeypatch, input_mode):
    import sys
    from dynet_py import cli

    source = tmp_path / "bad.csv"
    data = pd.DataFrame({"network": ["first", "second"], "from": ["A", "A"],
                         "to": ["B", "B"], "weight": [1, "invalid"]})
    data.to_csv(source, index=False)
    sources = [str(source)] if input_mode == "--input-csv" else [str(source), str(tmp_path / "good.csv")]
    data.iloc[:1].to_csv(tmp_path / "good.csv", index=False)
    destination = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["dynet-py", input_mode, *sources, "--out-dir", str(destination)])
    with pytest.raises(ValueError, match="Weights must be finite real numbers"):
        cli.main()
    assert not destination.exists()


def test_cli_does_not_silently_drop_missing_network_names(tmp_path, monkeypatch):
    import sys
    from dynet_py import cli

    source = tmp_path / "bad.csv"
    pd.DataFrame({"network": ["first", None], "from": ["A", "A"],
                  "to": ["B", "B"]}).to_csv(source, index=False)
    destination = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["dynet-py", "--input-csv", str(source), "--out-dir", str(destination)])
    with pytest.raises(ValueError, match="Network names must not contain missing values"):
        cli.main()
    assert not destination.exists()
