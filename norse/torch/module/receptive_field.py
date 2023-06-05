"""
These receptive fields are derived from scale-space theory, specifically in the paper `Normative theory of visual receptive fields by Lindeberg, 2021 <https://www.sciencedirect.com/science/article/pii/S2405844021000025>`_.

For use in spiking / binary signals, see the paper on `Translation and Scale Invariance for Event-Based Object tracking by Pedersen et al., 2023 <https://dl.acm.org/doi/10.1145/3584954.3584996>`_
"""

from typing import List, Tuple, Union

import torch

from norse.torch.module.leaky_integrator_box import LIBoxCell
from norse.torch.module.snn import SNNCell
from norse.torch.functional.receptive_field import (
    spatial_receptive_fields_with_derivatives,
)


class SpatialReceptiveField2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_scales: int,
        n_angles: int,
        n_ratios: int,
        size: int,
        derivatives: Union[int, List[Tuple[int, int]]] = 0,
        **kwargs
    ) -> None:
        """
        Creates a spatial receptive field as 2-dimensional convolutions.
        The parameters decide the number of combinations to scan over, i. e. the number of receptive fields to generate.
        Specifically, we generate ``n_scales * n_angles * (n_ratios - 1) + n_scales`` output_channels.

        The ``(n_ratios - 1) + n_scales`` terms exist because at ``ratio = 1``, fields are perfectly symmetrical, and there
        is therefore no reason to scan over the angles and scales for ``ratio = 1``.
        However, ``n_scales`` receptive field still needs to be added (one for each scale-space).

        Arguments:
          n_scales (int): Number of scaling combinations (the size of the receptive field) drawn from a logarithmic distribution
          n_angles (int): Number of angular combinations (the orientation of the receptive field)
          n_ratios (int): Number of eccentricity combinations (how "flat" the receptive field is)
          size (int): The size of the square kernel in pixels
          derivatives (Union[int, List[Tuple[int, int]]]): The number of derivatives to use in the receptive field.
          **kwargs: Arguments passed on to the underlying torch.nn.Conv2d
        """
        super().__init__()
        fields = spatial_receptive_fields_with_derivatives(
            n_scales, n_angles, n_ratios, size, derivatives
        )
        self.out_channels = len(fields.shape[0])
        self.conv = torch.nn.Conv2d(in_channels, fields.shape[0], size, **kwargs)
        self.conv.weight = torch.nn.Parameter(
            fields.unsqueeze(1).repeat(1, in_channels, 1, 1)
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class SpatioTemporalReceptiveField2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_scales: int,
        n_angles: int,
        n_ratios: int,
        n_times: int,
        size: int,
        activation: SNNCell = LIBoxCell,
    ):
        """Creates a spatio-temporal receptive field for 2-dimensional inputs."""
