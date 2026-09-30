import numpy as np
import pandas as pd
import pytest

from dynet_py import format_indata, prepare_networks, rewiring_analysis
from dynet_py.core import _PreparedNetwork


def _dataframe_reference(inputs, structure_only):
    """The previous pandas calculation, as a regression oracle for vectorization."""
    matrices = format_indata(inputs)
    nodes = pd.Index(dict.fromkeys(node for matrix in matrices for node in matrix.index))
    matrices = [matrix.reindex(index=nodes, columns=nodes, fill_value=0.) for matrix in matrices]
    if structure_only:
        matrices = [(matrix != 0).astype(float) for matrix in matrices]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        mean = sum(matrices) / sum((matrix != 0).astype(float) for matrix in matrices)
        mean = mean.replace([np.inf, -np.inf], np.nan).fillna(0.)
        standardized = [(matrix / mean).fillna(0.) for matrix in matrices]
        centroid = sum(standardized) / len(matrices)
        squared = [(matrix - centroid) ** 2 for matrix in standardized]
        distances = [matrix.sum(axis=0) + matrix.sum(axis=1) - np.diag(matrix)
                     for matrix in squared]
        rewiring = np.vstack(distances).sum(axis=0) / (len(matrices) - 1)
        union = sum((matrix > 0).astype(int) for matrix in matrices) > 0
        degree = (union.sum(axis=0) + union.sum(axis=1) - np.diag(union)).to_numpy(dtype=float)
        return pd.DataFrame({"name": nodes, "rewiring": rewiring, "degree": degree,
                             "degree_corrected_rewiring": rewiring / degree})


def _case_inputs(case):
    if case == "empty":
        return [np.zeros((0, 0)), pd.DataFrame(columns=["from", "to"])]
    if case == "isolates":
        return [pd.DataFrame(0., index=list("ABC"), columns=list("ABC")),
                pd.DataFrame(0., index=list("DC"), columns=list("DC"))]
    if case == "cancellation":
        return [np.array([[1., 2.], [0., 0.]]), np.array([[-1., -2.], [0., 0.]])]
    if case == "overflow":
        return [np.full((2, 2), 1e308), np.full((2, 2), 1e308)]
    if case == "mixed":
        return {
            "edges": pd.DataFrame({"from": ["B", "B", "A", "D"],
                                    "to": ["A", "A", "A", "D"], "weight": [1., 2., -1., 0.]}),
            "matrix": pd.DataFrame([[0., -2., 0.], [0., 1., 4.], [2., 0., 0.]],
                                   index=list("CAB"), columns=list("CAB")),
        }
    rng = np.random.default_rng(case)
    result = {}
    for i in range(4):
        nodes = rng.permutation(list("ABCDEFG"))[:i + 3]
        values = rng.integers(-2, 4, size=(len(nodes), len(nodes))).astype(float)
        result[str(i)] = pd.DataFrame(values, index=nodes, columns=nodes)
    return result


@pytest.mark.parametrize("case", ["empty", "isolates", "cancellation", "overflow", "mixed", 0, 1, 2, 3, 4])
@pytest.mark.parametrize("structure_only", [False, True])
def test_tensor_rewiring_matches_previous_calculation(case, structure_only):
    inputs = _case_inputs(case)
    expected = _dataframe_reference(inputs, structure_only)
    actual = rewiring_analysis(inputs, structure_only=structure_only)
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)


def test_tensor_aligns_labels_preserves_isolates_and_is_cached():
    edges = pd.DataFrame({"from": ["B", "D"], "to": ["A", "D"], "weight": [2., 0.]})
    matrix = pd.DataFrame([[0., 3.], [4., 0.]], index=list("CA"), columns=list("CA"))
    prepared = prepare_networks({"first": edges, "second": matrix})
    assert prepared.node_names == ("B", "D", "A", "C")
    tensor = prepared.adjacency_tensor()
    expected = np.zeros((2, 4, 4))
    expected[0, 0, 2] = 2.
    expected[1, 3, 2] = 3.
    expected[1, 2, 3] = 4.
    np.testing.assert_array_equal(tensor, expected)
    assert tensor.dtype == np.float64
    assert tensor is prepared.adjacency_tensor()
    assert not tensor.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        tensor[0, 0, 2] = 99.
    edges.loc[0, "weight"] = 99.
    matrix.iloc[0, 1] = 99.
    exported = tensor.copy()
    exported[:] = 0.
    for structural in (True, False, True):
        rewiring_analysis(prepared, structure_only=structural)
        assert tensor is prepared.adjacency_tensor()
        np.testing.assert_array_equal(tensor, expected)


def test_edge_list_rewiring_never_builds_individual_dataframe_matrices(monkeypatch):
    inputs = {
        "first": pd.DataFrame({"from": ["A"], "to": ["B"], "weight": [1.]}),
        "second": pd.DataFrame({"from": ["A"], "to": ["B"], "weight": [3.]}),
    }
    prepared = prepare_networks(inputs)

    def unexpected_matrix(self):
        raise AssertionError("Rewiring should populate the tensor directly from edges")

    monkeypatch.setattr(_PreparedNetwork, "adjacency", unexpected_matrix)
    result = rewiring_analysis(prepared)
    np.testing.assert_allclose(result["rewiring"], [0.5, 0.5])
    np.testing.assert_allclose(result["degree"], [1., 1.])
    np.testing.assert_allclose(result["degree_corrected_rewiring"], [0.5, 0.5])
    assert all(network._adjacency is None for network in prepared.values())


def test_self_loop_contribution_is_counted_once():
    inputs = np.array([[[1.]], [[3.]]])
    result = rewiring_analysis(inputs)
    assert result.loc[0, "name"] == "0"
    assert result.loc[0, "rewiring"] == 0.5
    assert result.loc[0, "degree"] == 1.


def test_too_few_networks_fail_before_allocating_tensor(monkeypatch):
    from dynet_py import PreparedNetworks

    def unexpected_tensor(self):
        raise AssertionError("The network count must be checked before allocating")

    monkeypatch.setattr(PreparedNetworks, "adjacency_tensor", unexpected_tensor)
    for inputs in ([], [np.zeros((2, 2))]):
        with pytest.raises(ValueError, match="at least two"):
            rewiring_analysis(inputs)
