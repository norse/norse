"""
A module for plotting event-based data and module information
such as weights and neuron parameters.
"""
from .plot import (
    plot_heatmap_2d,
    plot_heatmap_3d,
    plot_histogram_2d,
    plot_neuron_states,
    plot_scatter_3d,
    plot_spikes_2d,
    plot_izhikevich,
)

__all__ = [
    "plot_heatmap_2d",
    "plot_heatmap_3d",
    "plot_histogram_2d",
    "plot_neuron_states",
    "plot_scatter_3d",
    "plot_spikes_2d",
    "plot_izhikevich",
]
