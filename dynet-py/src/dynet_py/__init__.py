"""Use rewiring_analysis for scores and compare_conditions for edge/degree changes."""

from importlib.metadata import PackageNotFoundError, version

from .core import (
    PreparedNetworks,
    calculate_jaccard_indices,
    compare_targeting,
    compare_condition_pair,
    compare_conditions,
    plot_condition_changes,
    prepare_condition_data,
    rewiring_analysis,
    rewiring_plot,
    dynet_internal,
    dynet_main,
    dynet_plot,
    format_indata,
    prepare_networks,
    package_data,
    package_data_remap,
    package_data_rename,
    small_multiples_plot,
)

try:
    __version__ = version("dynet-py")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "PreparedNetworks",
    "format_indata",
    "prepare_networks",
    "rewiring_analysis",
    "rewiring_plot",
    "small_multiples_plot",
    "calculate_jaccard_indices",
    "compare_targeting",
    "prepare_condition_data",
    "compare_condition_pair",
    "compare_conditions",
    "plot_condition_changes",
    "package_data",
    "package_data_rename",
    "package_data_remap",
    "dynet_internal",
    "dynet_main",
    "dynet_plot",
]
