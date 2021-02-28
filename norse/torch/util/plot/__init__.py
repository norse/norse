"""
A module for plotting event-based data and module information
such as weights and neuron parameters.
"""
from .plot import (
    PlotParameters,
    plot_heatmap_2d,
    plot_heatmap_3d,
    plot_histogram_2d,
    plot_scatter_3d,
)

__all__ = [
    "PlotParameters",
    "plot_heatmap_2d",
    "plot_heatmap_3d",
    "plot_histogram_2d",
    "plot_scatter_3d",
]
