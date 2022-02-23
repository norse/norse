norse.torch.utils
=================

Utilities for spiking neural networks based on `PyTorch <https://pytorch.org>`_.

Packages and subpackages may depend on Matplotlib and Tensorboard.

.. contents:: norse.torch.utils
    :depth: 2
    :local:
    :backlinks: top


Plotting
--------

.. currentmodule:: norse.torch.utils.plot
.. autosummary::
    :toctree: generated
    :nosignatures:
    
    plot_heatmap_2d
    plot_heatmap_3d
    plot_histogram_2d
    plot_izhikevich
    plot_neuron_states
    plot_scatter_3d
    plot_spikes_2d

Tensorboard
-----------

.. currentmodule:: norse.torch.utils.tensorboard
.. autosummary::
    :toctree: generated
    :nosignatures:
    
    hook_spike_activity_mean
    hook_spike_activity_sum
    hook_spike_histogram_mean
    hook_spike_histogram_sum
    hook_spike_image