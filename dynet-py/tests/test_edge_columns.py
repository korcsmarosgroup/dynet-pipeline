from itertools import combinations
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dynet_py import (
    calculate_jaccard_indices, compare_targeting, format_indata, package_data,
    prepare_condition_data, prepare_networks, rewiring_analysis, rewiring_plot,
    small_multiples_plot,
)


ENDPOINTS = [("source", "target"), ("from", "to"), ("src", "dst")]


@pytest.mark.parametrize("endpoints", ENDPOINTS)
def test_endpoint_aliases_produce_identical_network_results(endpoints):
    canonical = pd.DataFrame({
        "source": ["A", "A", "B"], "target": ["B", "B", "C"], "weight": [1., 2., 4.],
    })
    alias = canonical.rename(columns=dict(zip(ENDPOINTS[0], endpoints)))
    before = alias.copy(deep=True)
    prepared = prepare_networks({"canonical": canonical, "alias": alias})
    matrices = format_indata(prepared)
    pd.testing.assert_frame_equal(matrices[0], matrices[1])
    assert matrices[1].loc["A", "B"] == 3.
    for backend in ("dense", "sparse"):
        assert rewiring_analysis(prepared, backend=backend)["rewiring"].eq(0).all()
    assert compare_targeting(prepared)["deltaTargeting"].eq(0).all()
    assert calculate_jaccard_indices(prepared).eq(1).all().all()
    scores = rewiring_analysis(prepared)
    for figure in (rewiring_plot(prepared, scores), small_multiples_plot(prepared, "A")):
        assert figure.axes
        plt.close(figure)
    pd.testing.assert_frame_equal(alias, before)


@pytest.mark.parametrize("endpoints", ENDPOINTS)
@pytest.mark.parametrize("prepare", [prepare_condition_data, package_data])
def test_condition_preparation_normalizes_aliases_without_mutation(endpoints, prepare):
    raw = pd.DataFrame({endpoints[0]: ["A", "A"], endpoints[1]: ["B", "B"],
                        "condition": ["T0", "T0"]}, index=[7, 9])
    before = raw.copy(deep=True)
    expected = pd.DataFrame({"condition": ["T0"], "source": ["A"],
                             "target": ["B"], "weight": [2.]})
    pd.testing.assert_frame_equal(prepare(raw), expected)
    pd.testing.assert_frame_equal(raw, before)


@pytest.mark.parametrize("pairs", list(combinations(ENDPOINTS, 2)))
def test_multiple_endpoint_pairs_are_rejected(pairs):
    raw = pd.DataFrame({name: [value] for pair in pairs for name, value in zip(pair, ["A", "B"])})
    raw["condition"] = "T0"
    for prepare in (lambda data: prepare_networks([data]), prepare_condition_data, package_data):
        with pytest.raises(ValueError, match="Ambiguous edge-list columns"):
            prepare(raw)


@pytest.mark.parametrize("endpoints", ENDPOINTS)
def test_duplicate_columns_are_rejected_for_all_aliases(endpoints):
    raw = pd.DataFrame([["A", "B", "C", "T0"]],
                       columns=[endpoints[0], endpoints[1], endpoints[1], "condition"])
    with pytest.raises(ValueError, match="column names must be unique"):
        prepare_networks([raw])
    with pytest.raises(ValueError, match="column names must be unique"):
        prepare_condition_data(raw)


@pytest.mark.parametrize("endpoints", ENDPOINTS)
def test_invalid_alias_weights_still_raise(endpoints):
    raw = pd.DataFrame({endpoints[0]: ["A"], endpoints[1]: ["B"],
                        "condition": ["T0"], "weight": [np.inf]})
    with pytest.raises(ValueError, match="Weights must be finite real numbers"):
        prepare_networks([raw])
    with pytest.raises(ValueError, match="Weights must be finite real numbers"):
        prepare_condition_data(raw)


def test_explicit_condition_columns_override_unused_canonical_columns():
    raw = pd.DataFrame({"source": ["unused"], "target": ["unused"],
                        "origin": ["A"], "destination": ["B"],
                        "condition": ["unused"], "time": ["T0"],
                        "weight": [999.], "value": [2.]})
    result = prepare_condition_data(raw, source="origin", target="destination",
                                    condition="time", weight="value")
    assert result.to_dict("records") == [
        {"condition": "T0", "source": "A", "target": "B", "weight": 2.},
    ]
    aliases = raw.rename(columns={"origin": "src", "destination": "dst"})
    selected = prepare_condition_data(aliases, source="src", target="dst",
                                      condition="time", weight="value")
    pd.testing.assert_frame_equal(result, selected)


@pytest.mark.parametrize("column", ["source", "target"])
def test_single_custom_endpoint_keeps_other_canonical_default(column):
    raw = pd.DataFrame({"source": ["A"], "target": ["B"], "condition": ["T0"]})
    custom = raw.rename(columns={column: "custom"})
    pd.testing.assert_frame_equal(prepare_condition_data(custom, **{column: "custom"}),
                                  prepare_condition_data(raw))


@pytest.mark.parametrize("labels", ENDPOINTS)
def test_adjacency_node_labels_can_match_endpoint_names(labels):
    matrix = pd.DataFrame([[0., 2.], [3., 0.]], index=labels, columns=labels)
    inputs = [matrix.iloc[::-1], matrix]
    pd.testing.assert_frame_equal(format_indata(inputs)[0], matrix)
    for backend in ("dense", "sparse"):
        assert rewiring_analysis(inputs, backend=backend)["rewiring"].eq(0).all()


@pytest.mark.parametrize("input_mode", ["--input-csv", "--edge-lists"])
def test_cli_rejects_ambiguous_aliases_before_writing(tmp_path, monkeypatch, input_mode):
    from dynet_py import cli

    raw = pd.DataFrame({"network": ["first", "second"], "source": ["A", "A"],
                        "target": ["B", "B"], "src": ["C", "C"], "dst": ["D", "D"]})
    path = tmp_path / "ambiguous.csv"
    raw.to_csv(path, index=False)
    paths = [str(path)] if input_mode == "--input-csv" else [str(path), str(path)]
    destination = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["dynet-py", input_mode, *paths, "--out-dir", str(destination)])
    with pytest.raises(ValueError, match="Ambiguous edge-list columns"):
        cli.main()
    assert not destination.exists()
