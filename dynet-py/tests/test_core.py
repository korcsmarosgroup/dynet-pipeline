import pandas as pd

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
    out = dynet_main(package_data(raw), conditions=["T0", "T1"])
    comp = out["comparisons"][0]
    statuses = comp["edge_changes"]["status"].value_counts().to_dict()
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
    out = dynet_main(package_data(raw), pairwise="all")
    assert len(out["comparisons"]) == 3


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
