import numpy as np
import pandas as pd
import pytest

from dynet_py import PreparedNetworks, prepare_networks, rewiring_analysis
from dynet_py.core import _PreparedNetwork


def test_sparse_rewiring_counts_missing_observations_as_zero():
    present = pd.DataFrame({"from": ["A"], "to": ["B"], "weight": [2.]})
    empty = pd.DataFrame(columns=["from", "to"])
    result = rewiring_analysis([present, empty, empty], backend="sparse")
    # Standardized observations [1, 0, 0] have sample variance 1/3.
    np.testing.assert_allclose(result["rewiring"], [1 / 3, 1 / 3])
    np.testing.assert_array_equal(result["degree"], [1., 1.])


@pytest.mark.parametrize("backend", ["sparse", "auto"])
def test_large_sparse_inputs_never_allocate_dense_matrices(monkeypatch, backend):
    nodes = np.arange(10000)
    edges = pd.DataFrame({"from": nodes, "to": nodes + 1, "weight": 1.})
    prepared = prepare_networks([edges, edges, edges])

    def unexpected_dense(self):
        raise AssertionError("Sparse rewiring must not allocate adjacency matrices")

    monkeypatch.setattr(PreparedNetworks, "adjacency_tensor", unexpected_dense)
    monkeypatch.setattr(_PreparedNetwork, "adjacency", unexpected_dense)
    result = rewiring_analysis(prepared, backend=backend)
    assert len(result) == 10001
    assert result["rewiring"].eq(0).all()
    assert prepared._adjacency_tensor is None
    assert all(network._adjacency is None for network in prepared.values())


def test_sparse_coordinates_are_cached_without_mutating_weights(monkeypatch):
    edges = pd.DataFrame({"from": ["A", "B"], "to": ["B", "A"], "weight": [-2., 3.]})
    prepared = prepare_networks([edges, edges])
    baseline = rewiring_analysis(prepared, backend="sparse")
    coordinates = prepared._sparse_edges
    weights = coordinates.weights.copy()

    def unexpected_edges(self):
        raise AssertionError("Sparse coordinates should be reused")

    monkeypatch.setattr(_PreparedNetwork, "edges", unexpected_edges)
    rewiring_analysis(prepared, backend="sparse", structure_only=True)
    pd.testing.assert_frame_equal(rewiring_analysis(prepared, backend="sparse"), baseline)
    assert prepared._sparse_edges is coordinates
    np.testing.assert_array_equal(coordinates.weights, weights)
    assert not coordinates.weights.flags.writeable


@pytest.mark.parametrize("values", [
    [1., 1. + 1e-12, 1. - 1e-12],
    [1e-300, 2e-300, 0.],
    [1., -1., 0.],
    [1e308, 1e308, 0.],
    [1e-323, 0., 0., 0., 0.],
    [1e300, -1e300, 1e-10],
])
@pytest.mark.parametrize("self_loop", [False, True])
def test_sparse_extreme_weights_match_dense(values, self_loop):
    target = "A" if self_loop else "B"
    inputs = [pd.DataFrame({"from": ["A"], "to": [target], "weight": [value]}) for value in values]
    expected = rewiring_analysis(inputs, backend="dense")
    actual = rewiring_analysis(inputs, backend="sparse")
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-30)


def test_backend_validation():
    with pytest.raises(ValueError, match="backend must be"):
        rewiring_analysis([], backend="unknown")


@pytest.mark.parametrize("nodes, edges_per_network, expected", [
    (10, 1, "dense"),
    (250, 6250, "sparse"),  # Exactly 10% of the full set of possible edges.
    (250, 6251, "dense"),
])
def test_auto_selects_by_size_and_density(nodes, edges_per_network, expected):
    labels = np.arange(nodes)
    ids = np.arange(edges_per_network)
    frame = pd.DataFrame({
        "from": np.concatenate((ids // nodes, labels)),
        "to": np.concatenate((ids % nodes, labels)),
        "weight": np.concatenate((np.ones(len(ids)), np.zeros(nodes))),
    })
    prepared = prepare_networks([frame, frame])
    actual = rewiring_analysis(prepared)
    assert prepared._auto_backend == expected
    assert (prepared._adjacency_tensor is None) == (expected == "sparse")
    pd.testing.assert_frame_equal(actual, rewiring_analysis(prepared, backend=expected))


@pytest.mark.parametrize("backend", ["sparse", "dense", "auto"])
def test_cli_backend_matches_api(tmp_path, monkeypatch, backend):
    import sys
    import matplotlib.pyplot as plt
    from dynet_py import cli

    data = pd.DataFrame({"network": ["first", "second"], "from": ["A", "A"],
                         "to": ["B", "B"], "weight": [1., 3.]})
    source = tmp_path / "networks.csv"
    data.to_csv(source, index=False)
    destination = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["dynet-py", "--input-csv", str(source),
                                     "--backend", backend, "--out-dir", str(destination)])
    cli.main()
    result = pd.read_csv(destination / "dynet_py_output.csv")
    np.testing.assert_allclose(result["rewiring"], [0.5, 0.5])
    assert {file.name for file in destination.iterdir()} == {
        "dynet_py_output.csv", "compare_targeting.csv", "dynet_py_plot.png", "small_multiples_plot.png",
    }
    plt.close("all")
