"""
Functional components for measuring synaptic operations (SynOps).

A synaptic operation occurs when a spike propagates through a synapse. For a
linear layer, each firing neuron triggers one multiply-accumulate per output
neuron. For a convolutional layer, each firing activation triggers one MAC per
output channel per kernel element (divided by the number of groups).

This metric is commonly used in neuromorphic computing to quantify computational
cost and energy efficiency of SNNs.

Reference: Pfeiffer, M. & Pfeil, T. (2018). Deep learning with spiking neurons:
Opportunities and challenges. Frontiers in Computational Neuroscience.
"""

from typing import Optional, Union

import torch
from torch import nn


def linear_synops_accumulator(
    z: torch.Tensor,
    layer: nn.Linear,
    state: Optional[Union[int, torch.Tensor]] = None,
) -> Union[int, torch.Tensor]:
    """
    Count synaptic operations for a linear layer.

    Each firing input neuron triggers ``out_features`` multiply-accumulate
    operations (one per output neuron).

    Parameters:
        z (torch.Tensor): Spike tensor from the preceding spiking layer.
        layer (nn.Linear): The linear (synaptic) layer that receives the spikes.
        state (Optional[Union[int, torch.Tensor]]): Accumulated synops from previous
            steps. Defaults to 0.

    Returns:
        The updated synops count ``state + spike_count * out_features``.
    """
    if state is None:
        state = 0
    return state + z.sum() * layer.out_features


def conv_synops_accumulator(
    z: torch.Tensor,
    layer: nn.Conv2d,
    state: Optional[Union[int, torch.Tensor]] = None,
) -> Union[int, torch.Tensor]:
    """
    Count synaptic operations for a 2-D convolutional layer.

    Each firing activation at any spatial position triggers
    ``out_channels * kernel_h * kernel_w / groups`` multiply-accumulate
    operations.

    Parameters:
        z (torch.Tensor): Spike tensor from the preceding spiking layer.
        layer (nn.Conv2d): The convolutional (synaptic) layer that receives the spikes.
        state (Optional[Union[int, torch.Tensor]]): Accumulated synops from previous
            steps. Defaults to 0.

    Returns:
        The updated synops count.
    """
    if state is None:
        state = 0
    kH, kW = (
        layer.kernel_size
        if isinstance(layer.kernel_size, tuple)
        else (layer.kernel_size, layer.kernel_size)
    )
    fan_out = layer.out_channels * kH * kW // layer.groups
    return state + z.sum() * fan_out


def synops_accumulator(
    z: torch.Tensor,
    layer: nn.Module,
    state: Optional[Union[int, torch.Tensor]] = None,
) -> Union[int, torch.Tensor]:
    """
    Count synaptic operations for a supported layer type (Linear or Conv2d).

    Parameters:
        z (torch.Tensor): Spike tensor from the preceding spiking layer.
        layer (nn.Module): The synaptic layer (``nn.Linear`` or ``nn.Conv2d``).
        state (Optional[Union[int, torch.Tensor]]): Accumulated synops from previous
            steps. Defaults to 0.

    Returns:
        The updated synops count.

    Raises:
        TypeError: If ``layer`` is not a supported type.
    """
    if isinstance(layer, nn.Linear):
        return linear_synops_accumulator(z, layer, state)
    if isinstance(layer, nn.Conv2d):
        return conv_synops_accumulator(z, layer, state)
    raise TypeError(
        f"synops_accumulator does not support layer type {type(layer).__name__}. "
        "Supported types: nn.Linear, nn.Conv2d."
    )
