from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from dynet_py import (
    compare_condition_pair, compare_conditions, dynet_internal, dynet_main,
    dynet_plot, plot_condition_changes, prepare_condition_data,
)


def _conditions():
    return prepare_condition_data(pd.DataFrame({
        "source": ["A", "C", "A", "B", "B", "D"],
        "target": ["B", "C", "B", "C", "C", "A"],
        "condition": ["T0", "T0", "T1", "T1", "T2", "T2"],
        "weight": [1., 0., 3., 2., 5., 1.],
    }), drop_self_loops=False)


@pytest.mark.parametrize("output", ["nodes", "edges", "summary"])
@pytest.mark.parametrize("batch, pair", [(compare_conditions, compare_condition_pair), (dynet_main, dynet_internal)])
def test_flat_tables_have_stable_columns_and_match_single_pairs(output, batch, pair):
    data = _conditions()
    before = data.copy(deep=True)
    result = batch(data, pairwise="all", output=output)
    assert isinstance(result, pd.DataFrame)
    assert result.columns.is_unique
    assert list(result.columns[:3]) == ["comparison", "condition_a", "condition_b"]
    assert not any("T0" in column or "T1" in column or "T2" in column for column in result.columns)
    assert not result.isna().any().any()
    for a, b in [("T0", "T1"), ("T0", "T2"), ("T1", "T2")]:
        selected = result[(result["condition_a"] == a) & (result["condition_b"] == b)].reset_index(drop=True)
        pd.testing.assert_frame_equal(selected, pair(data, a, b, output=output))
    restored = pd.read_csv(StringIO(result.to_csv(index=False)))
    pd.testing.assert_frame_equal(restored, result, check_dtype=False)
    pd.testing.assert_frame_equal(data, before)


def test_tables_report_expected_counts_weights_and_degree_changes():
    data = _conditions()
    nodes = compare_conditions(data)
    selected = nodes.query("condition_a == 'T0' and condition_b == 'T1'").set_index("node")
    assert selected["degree_change_score"].to_dict() == {"B": 5., "C": 3., "A": 2.}
    assert selected.loc["C", "degree_a"] == 2  # Retained zero-weight loop counts twice.
    assert selected.loc["B", "weight_degree_b"] == 5
    edges = compare_condition_pair(data, "T0", "T1", output="edges").set_index(["source", "target"])
    assert edges.loc[("A", "B"), ["status", "weight_a", "weight_b", "delta_weight"]].tolist() == ["kept", 1., 3., 2.]
    assert edges.loc[("C", "C"), "status"] == "lost"
    summary = compare_conditions(data, output="summary")
    assert summary[["n_kept", "n_lost", "n_gained", "n_nodes"]].values.tolist() == [[1, 1, 1, 3], [1, 1, 1, 4]]


@pytest.mark.parametrize("labels", [("before", "after"), ("change_score", "a"), ("b", "a")])
def test_condition_names_do_not_change_measurement_columns(labels):
    raw = pd.DataFrame({"source": ["A", "A"], "target": ["B", "B"],
                        "condition": labels, "weight": [1., 3.]})
    result = compare_condition_pair(prepare_condition_data(raw), *labels)
    assert result.columns.is_unique
    assert result["degree_a"].eq(1).all()
    assert result["weight_degree_a"].eq(1).all()
    assert result["weight_degree_b"].eq(3).all()
    assert result["degree_change_score"].eq(2).all()


@pytest.mark.parametrize("analyze, args", [
    (compare_conditions, ()), (dynet_main, ()),
    (compare_condition_pair, ("T0", "T1")), (dynet_internal, ("T0", "T1")),
])
def test_invalid_output_is_rejected(analyze, args):
    with pytest.raises(ValueError, match="output must be"):
        analyze(_conditions(), *args, output="unknown")


@pytest.mark.parametrize("analyze, args", [(dynet_main, ()), (dynet_internal, ("T0", "T1"))])
def test_legacy_score_option_changes_only_flat_node_column_name(analyze, args):
    current = analyze(_conditions(), *args)
    legacy_name = analyze(_conditions(), *args, legacy_score_name=True)
    pd.testing.assert_frame_equal(current.rename(columns={"degree_change_score": "rewiring_score"}), legacy_name)


@pytest.mark.parametrize("output", ["nodes", "edges", "summary"])
@pytest.mark.parametrize("plot", [plot_condition_changes, dynet_plot])
def test_plot_infers_table_kind_without_mutation(output, plot):
    result = compare_conditions(_conditions(), output=output)
    before = result.copy(deep=True)
    figure = plot(result, comparison=-1)
    axis = figure.axes[0]
    assert "T1_vs_T2" in axis.get_title()
    if output == "nodes":
        assert axis.get_xlabel() == "Degree-change score"
    else:
        assert [bar.get_height() for bar in axis.patches] == [1, 1, 1]
    pd.testing.assert_frame_equal(result, before)
    plt.close(figure)


@pytest.mark.parametrize("analyze, args", [(dynet_main, ()), (dynet_internal, ("T0", "T1"))])
@pytest.mark.parametrize("legacy_score_name", [False, True])
def test_plots_still_accept_legacy_dictionaries(analyze, args, legacy_score_name):
    result = analyze(_conditions(), *args, output="legacy", legacy_score_name=legacy_score_name)
    pair = result["comparisons"][0] if "comparisons" in result else result
    before = pair["node_changes"].copy(deep=True)
    assert "weight_T0" in pair["edge_changes"]
    assert "degree_T0" in before
    assert "out_weight_x" in before
    for what in (None, "nodes"):
        figure = dynet_plot(result, what=what)
        plt.close(figure)
    pd.testing.assert_frame_equal(pair["node_changes"], before)


def test_plot_distinguishes_pairs_with_identical_display_names():
    labels = ["a", "b_vs_c", "a_vs_b", "c"]
    raw = pd.DataFrame([(label + str(i), "target", label) for label, count in zip(labels, [2, 1, 3, 1])
                        for i in range(count)], columns=["source", "target", "condition"])
    summary = compare_conditions(prepare_condition_data(raw), conditions=labels, pairwise="all", output="summary")
    assert summary.iloc[0]["comparison"] == summary.iloc[-1]["comparison"]
    for comparison, lost in ((0, 2), (-1, 3)):
        figure = plot_condition_changes(summary, comparison=comparison)
        assert [bar.get_height() for bar in figure.axes[0].patches] == [1, lost, 0]
        plt.close(figure)


@pytest.mark.parametrize("output, what", [("nodes", "edges"), ("edges", "nodes"), ("summary", "nodes")])
def test_plot_reports_incompatible_table(output, what):
    with pytest.raises(ValueError, match="plots require output="):
        plot_condition_changes(compare_conditions(_conditions(), output=output), what=what)
