import sys

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import pandas as pd
import pytest

from dynet_py import cli, rewiring_analysis


def _csv(tmp_path):
    raw = pd.DataFrame({
        "network": ["first"] * 3 + ["second"] * 3,
        "source": ["A", "B", "C", "A", "C", "D"],
        "target": ["B", "C", "D", "C", "D", "B"],
        "weight": [1., 2., 1., 3., 2., 1.],
    })
    path = tmp_path / "networks.csv"
    raw.to_csv(path, index=False)
    return path, raw


def test_cli_plot_filters_keep_full_score_output(tmp_path, monkeypatch):
    path, raw = _csv(tmp_path)
    expected = rewiring_analysis({name: frame.drop(columns="network")
                                 for name, frame in raw.groupby("network", sort=False)})
    destination = tmp_path / "results"
    plots = []
    for name in ("rewiring_plot", "small_multiples_plot"):
        original = getattr(cli, name)

        def capture(*args, _original=original, **kwargs):
            figure = _original(*args, **kwargs)
            axes = [ax for ax in figure.axes if ax.get_label() != "<colorbar>"]
            for ax in axes:
                scatter = next(item for item in ax.collections if isinstance(item, PathCollection))
                assert len(scatter.get_offsets()) == 2
                assert not any(text.get_text() for text in ax.texts)
            plots.append(figure)
            return figure

        monkeypatch.setattr(cli, name, capture)
    monkeypatch.setattr(sys, "argv", [
        "dynet-py", "--input-csv", str(path), "--out-dir", str(destination),
        "--plot-layout", "circular", "--plot-labels", "none", "--plot-top-n", "2",
        "--plot-focus-node", "A", "--plot-radius", "2", "--panel-max-nodes", "2", "--plot-seed", "7",
    ])
    cli.main()
    pd.testing.assert_frame_equal(pd.read_csv(destination / "dynet_py_output.csv"), expected)
    assert len(plots) == 2
    assert all(not plt.fignum_exists(figure.number) for figure in plots)
    assert (destination / "dynet_py_plot.png").stat().st_size > 0
    assert (destination / "small_multiples_plot.png").stat().st_size > 0


@pytest.mark.parametrize("flag, value", [("--plot-top-n", "0"), ("--panel-max-nodes", "-1"),
                                        ("--plot-radius", "-1"), ("--plot-seed", "-1")])
def test_cli_invalid_plot_counts_are_rejected_before_outputs(tmp_path, monkeypatch, flag, value):
    destination = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["dynet-py", "--input-csv", "unused.csv",
                                     "--out-dir", str(destination), flag, value])
    with pytest.raises(SystemExit):
        cli.main()
    assert not destination.exists()


def test_cli_unknown_plot_focus_does_not_write_partial_outputs(tmp_path, monkeypatch):
    path, _ = _csv(tmp_path)
    destination = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["dynet-py", "--input-csv", str(path),
                                     "--out-dir", str(destination), "--plot-focus-node", "missing"])
    with pytest.raises(ValueError, match="Unknown focus node"):
        cli.main()
    assert not destination.exists()
