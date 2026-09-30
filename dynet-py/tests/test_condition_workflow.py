import matplotlib.pyplot as plt
import pandas as pd
import pytest

from dynet_py import (
    compare_condition_pair,
    compare_conditions,
    dynet_internal,
    dynet_main,
    dynet_plot,
    package_data,
    plot_condition_changes,
    prepare_condition_data,
    prepare_networks,
    rewiring_analysis,
)


def _swapped_neighbors():
    return pd.DataFrame({
        "source": ["A", "C", "A", "C"],
        "target": ["B", "D", "D", "B"],
        "condition": ["T0", "T0", "T1", "T1"],
        "weight": [1., 1., 1., 1.],
    })


def test_degree_change_and_standardized_rewiring_are_distinct_metrics():
    data = prepare_condition_data(_swapped_neighbors())
    changes = compare_condition_pair(data, "T0", "T1")
    assert changes["summary"]["n_gained"] == 2
    assert changes["summary"]["n_lost"] == 2
    assert changes["node_changes"]["degree_change_score"].eq(0).all()
    assert "rewiring_score" not in changes["node_changes"]

    networks = {
        name: frame.rename(columns={"source": "from", "target": "to"})
        for name, frame in data.groupby("condition")
    }
    scores = rewiring_analysis(prepare_networks(networks)).set_index("name")
    assert scores["rewiring"].to_dict() == {"A": 1., "B": 1., "C": 1., "D": 1.}


def test_pair_comparison_score_and_direction():
    data = prepare_condition_data(pd.DataFrame({
        "source": ["A", "A"], "target": ["B", "B"],
        "condition": ["before", "after"], "weight": [1., 3.],
    }))
    result = compare_condition_pair(data, "before", "after")
    assert result["edge_changes"]["status"].tolist() == ["kept"]
    assert result["edge_changes"]["delta_weight"].tolist() == [2.]
    assert result["node_changes"]["delta_degree"].eq(0).all()
    assert result["node_changes"]["delta_weight_degree"].eq(2).all()
    assert result["node_changes"]["degree_change_score"].eq(2).all()


def test_legacy_entry_points_preserve_result_schema_and_values():
    raw = _swapped_neighbors()
    pd.testing.assert_frame_equal(package_data(raw), prepare_condition_data(raw))
    data = prepare_condition_data(raw)
    current = compare_conditions(data)
    legacy = dynet_main(data)
    pd.testing.assert_frame_equal(current["edge_counts"], legacy["edge_counts"])
    for new_pair, old_pair in zip(current["comparisons"], legacy["comparisons"]):
        assert old_pair["summary"] == new_pair["summary"]
        assert old_pair["comparison"] == new_pair["comparison"]
        pd.testing.assert_frame_equal(old_pair["edge_changes"], new_pair["edge_changes"])
        pd.testing.assert_frame_equal(old_pair["node_changes"], new_pair["node_changes"].rename(
            columns={"degree_change_score": "rewiring_score"}))
    pair = dynet_internal(data, "T0", "T1")
    pd.testing.assert_frame_equal(pair["node_changes"], legacy["comparisons"][0]["node_changes"])


@pytest.mark.parametrize("pairwise, expected", [
    ("adjacent", ["T2_vs_T0", "T0_vs_T1"]),
    ("all", ["T2_vs_T0", "T2_vs_T1", "T0_vs_T1"]),
])
def test_condition_comparisons_respect_explicit_order(pairwise, expected):
    data = prepare_condition_data(pd.DataFrame({
        "source": ["A", "A", "A"], "target": ["B", "B", "B"],
        "condition": ["T0", "T1", "T2"],
    }))
    result = compare_conditions(data, conditions=["T2", "T0", "T1"], pairwise=pairwise)
    assert [pair["comparison"] for pair in result["comparisons"]] == expected


@pytest.mark.parametrize("bad_weight", ["bad", None, float("inf")])
def test_new_condition_preparation_rejects_invalid_weights(bad_weight):
    raw = _swapped_neighbors().astype({"weight": object})
    raw.loc[0, "weight"] = bad_weight
    with pytest.raises(ValueError, match="Weights must be finite real numbers"):
        prepare_condition_data(raw)


def test_legacy_preparation_retains_historical_coercion():
    raw = _swapped_neighbors().astype({"weight": object})
    raw.loc[0, "weight"] = "bad"
    legacy = package_data(raw)
    assert legacy.loc[(legacy["condition"] == "T0") & (legacy["source"] == "A"), "weight"].iloc[0] == 0.


def test_condition_comparison_rejects_network_collection_with_useful_error():
    with pytest.raises(ValueError, match="prepare_condition_data.*rewiring_analysis"):
        compare_conditions(prepare_networks([]))


@pytest.mark.parametrize("conditions", [["T0", "missing"], ["T0", "T0"], ["T0"]])
def test_condition_comparison_rejects_invalid_selection(conditions):
    with pytest.raises(ValueError):
        compare_conditions(prepare_condition_data(_swapped_neighbors()), conditions=conditions)


@pytest.mark.parametrize("plotter, analyze", [
    (plot_condition_changes, compare_conditions), (dynet_plot, dynet_main),
])
@pytest.mark.parametrize("what", ["nodes", "edges"])
def test_comparison_plots_use_clear_labels_without_mutating_results(plotter, analyze, what):
    result = analyze(prepare_condition_data(_swapped_neighbors()))
    before = result["comparisons"][0]["node_changes"].copy(deep=True)
    figure = plotter(result, what=what)
    assert "Rewir" not in figure.axes[0].get_title()
    if what == "nodes":
        assert figure.axes[0].get_xlabel() == "Degree-change score"
    pd.testing.assert_frame_equal(result["comparisons"][0]["node_changes"], before)
    plt.close(figure)
