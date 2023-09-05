"""
These receptive fields are derived from scale-space theory, specifically in the paper `Normative theory of visual receptive fields by Lindeberg, 2021 <https://www.sciencedirect.com/science/article/pii/S2405844021000025>`_.

For use in spiking / binary signals, see the paper on `Translation and Scale Invariance for Event-Based Object tracking by Pedersen et al., 2023 <https://dl.acm.org/doi/10.1145/3584954.3584996>`_
"""

from typing import Callable, List, NamedTuple, Optional, Tuple, Type, Union

import torch

from norse.torch.module.leaky_integrator_box import LIBoxCell, LIBoxParameters
from norse.torch.module.snn import SNNCell
from norse.torch.functional.receptive_field import (
    spatial_receptive_fields_with_derivatives,
    temporal_scale_distribution,
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
        min_scale: float = 0.2,
        max_scale: float = 1.5,
        min_ratio: float = 0.2,
        max_ratio: float = 1,
        aggregate: bool = True,
        **kwargs
    ) -> None:
        """
        Creates a spatial receptive field as 2-dimensional convolutions.
        The parameters decide the number of combinations to scan over, i. e. the number of receptive fields to generate.
        Specifically, we generate ``n_scales * n_angles * (n_ratios - 1) + n_scales`` output_channels with aggregation,
        and ``in_channels * (n_scales * n_angles * (n_ratios - 1) + n_scales)`` without aggregation.

        The ``(n_ratios - 1) + n_scales`` terms exist because at ``ratio = 1``, fields are perfectly symmetrical, and there
        is therefore no reason to scan over the angles and scales for ``ratio = 1``.
        However, ``n_scales`` receptive field still needs to be added (one for each scale-space).

        Parameters:
            n_scales (int): Number of scaling combinations (the size of the receptive field) drawn from a logarithmic distribution
            n_angles (int): Number of angular combinations (the orientation of the receptive field)
            n_ratios (int): Number of eccentricity combinations (how "flat" the receptive field is)
            size (int): The size of the square kernel in pixels
            derivatives (Union[int, List[Tuple[int, int]]]): The number of derivatives to use in the receptive field.
            aggregate (bool): If True, sums the input channels over all output channels. If False, every
                output channel is mapped to every input channel, which may blow up in complexity.
            **kwargs: Arguments passed on to the underlying torch.nn.Conv2d
        """
        super().__init__()
        fields = spatial_receptive_fields_with_derivatives(
            n_scales,
            n_angles,
            n_ratios,
            size,
            derivatives,
            min_scale,
            max_scale,
            min_ratio,
            max_ratio,
        )
        if aggregate:
            self.out_channels = fields.shape[0]
            weights = fields.unsqueeze(1).repeat(1, in_channels, 1, 1)
        else:
            self.out_channels = fields.shape[0] * in_channels
            empty_weights = torch.zeros(in_channels, fields.shape[0], size, size)
            weights = []
            for i in range(in_channels):
                in_weights = empty_weights.clone()
                in_weights[i] = fields
                weights.append(in_weights)
            weights = torch.concat(weights, 1).permute(1, 0, 2, 3)

        self.conv = torch.nn.Conv2d(in_channels, self.out_channels, size, **kwargs)
        self.conv.weight = torch.nn.Parameter(weights)

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class TemporalReceptiveField(torch.nn.Module):
    def __init__(
        self,
        shape: torch.Size,
        n_scales: int = 4,
        activation: Type[SNNCell] = LIBoxCell,
        activation_state_map: Callable[
            [torch.Tensor], NamedTuple
        ] = lambda t: LIBoxParameters(tau_mem_inv=t),
        min_scale: float = 1,
        max_scale: float = 1,
        c: float = 1.41421,
        time_constants: Optional[torch.Tensor] = None,
        dt: float = 0.001,
    ):
        """Creates ``n_scales`` temporal receptive fields for arbitrary n-dimensional inputs.
        The scale spaces are selected in a range of [min_scale, max_scale] using an exponential distribution, scattered using ``torch.linspace``.

        Parameters:
            shape (torch.Size): The shape of the incoming tensor, where the first dimension denote channels
            n_scales (int): The number of temporal scale spaces to iterate over.
            activation (SNNCell): The activation neuron. Defaults to LIBoxCell
            activation_state_map (Callable): A function that takes a tensor and provides a neuron parameter tuple.
                Required if activation is changed, since the default behaviour provides LIBoxParameters.
            min_scale (float): The minimum scale space. Defaults to 1.
            max_scale (Optional[float]): The maximum scale, given the growth parameter c. Defaults to 1.
            c (Optional[float]): The base from which to generate scale values. Should be a value between 1 to 2 exclusive. Defaults to sqrt(2).
            time_constants (Optional[torch.Tensor]): Hardcoded time constants. Will overwrite the automatically generated, logarithmically distributed scales, if set. Defaults to None.
            dt (float): Neuron simulation timestep. Defaults to 0.001.
        """
        super().__init__()
        if time_constants is None:
            taus = temporal_scale_distribution(n_scales, min_scale=min_scale) / dt
            self.time_constants = torch.stack(
                [
                    torch.full(
                        [shape[0], *[1 for i in range(len(shape) - 1)]],
                        tau,
                        dtype=torch.float32,
                    )
                    for tau in taus
                ]
            )
        else:
            self.time_constants = time_constants
        self.ps = torch.nn.Parameter(self.time_constants)
        # pytype: disable=missing-parameter
        self.neurons = activation(p=activation_state_map(self.ps), dt=dt)
        # pytype: enable=missing-parameter
        self.rf_dimension = len(shape)
        self.n_scales = n_scales

    def forward(self, x: torch.Tensor, state: Optional[NamedTuple] = None):
        x_repeated = torch.stack(
            [x for _ in range(self.n_scales)], dim=-self.rf_dimension - 1
        )
        return self.neurons(x_repeated, state)
