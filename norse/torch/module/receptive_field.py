"""
These receptive fields are derived from scale-space theory, specifically in the paper `Normative theory of visual receptive fields by Lindeberg, 2021 <https://www.sciencedirect.com/science/article/pii/S2405844021000025>`_.

For use in spiking / binary signals, see the paper on `Translation and Scale Invariance for Event-Based Object tracking by Pedersen et al., 2023 <https://dl.acm.org/doi/10.1145/3584954.3584996>`_
"""

from typing import Callable, NamedTuple, Optional, Tuple

import torch

from norse.torch.module.snn import SNNCell
from norse.torch.module.leaky_integrator_box import LIBoxCell, LIBoxParameters
from norse.torch.functional.receptive_field import (
    spatial_receptive_fields_with_derivatives,
    spatial_parameters,
    temporal_scale_distribution,
)


class SpatialReceptiveField2d(torch.nn.Module):
    """Creates a spatial receptive field as 2-dimensional convolutions.
    The `rf_parameters` are a tensor of shape `(n, 5)` where `n` is the number of receptive fields.
    If the `optimize_fields` flag is set to `True`, the `rf_parameters` will be optimized during training.

    Example:
        >>> import torch
        >>> from norse.torch import SpatialReceptiveField2d
        >>> parameters = torch.tensor([[1., 1., 1., 0., 0.]])
        >>> m = SpatialReceptiveField2d(1, 9, parameters)
        >>> m.weights.shape
        torch.Size([1, 1, 9, 9])
        >>> y = m(torch.empty(1, 1, 9, 9))
        >>> y.shape
        torch.Size([1, 1, 1, 1])

    Arguments:
        in_channels (int): Number of input channels
        size (int): Size of the receptive field
        rf_parameters (torch.Tensor): Parameters for the receptive fields in the order (scale, angle, ratio, dx, dy)
        aggregate (bool): If `True`, the receptive fields will be aggregated across channels. Defaults to `True`.
        domain (float): The domain of the receptive field. Defaults to `8`.
        optimize_fields (bool): If `True`, the `rf_parameters` will be optimized during training. Defaults to `True`.
        **kwargs: Additional arguments for the `torch.nn.functional.conv2d` function.
    """

    def __init__(
        self,
        in_channels: int,
        size: int,
        rf_parameters: torch.Tensor,
        aggregate: bool = True,
        domain: float = 8,
        optimize_fields: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.rf_parameters = (
            torch.nn.Parameter(rf_parameters) if optimize_fields else rf_parameters
        )
        self.rf_parameters_previous = torch.zeros_like(self.rf_parameters)

        self.in_channels = in_channels
        self.size = size
        self.aggregate = aggregate
        self.domain = domain
        self.out_channels = (
            rf_parameters.shape[0]
            if aggregate
            else rf_parameters.shape[0] * in_channels
        )
        self.kwargs = kwargs
        if "bias" not in self.kwargs:
            self.kwargs["bias"] = None

        # Register update and update the fields for the first time
        self.has_updated = True
        self._update_weights()

        if optimize_fields:

            def update_hook(m, gi, go):
                m.has_updated = True

            self.register_full_backward_hook(update_hook)

    def _update_weights(self):
        if self.has_updated:
            if not torch.all(torch.eq(self.rf_parameters_previous, self.rf_parameters)):
                # Reset the flag
                self.has_updated = False
                self.rf_parameters_previous = self.rf_parameters.detach().clone()
                # Calculate new weights
                fields = spatial_receptive_fields_with_derivatives(
                    self.rf_parameters,
                    size=self.size,
                    domain=self.domain,
                )
                if self.aggregate:
                    self.out_channels = fields.shape[0]
                    self.weights = fields.unsqueeze(1).repeat(1, self.in_channels, 1, 1)
                else:
                    self.out_channels = self.fields.shape[0] * self.in_channels
                    empty_weights = torch.zeros(
                        self.in_channels,
                        fields.shape[0],
                        self.size,
                        self.size,
                        device=self.rf_parameters.device,
                    )
                    weights = []
                    for i in range(self.in_channels):
                        in_weights = empty_weights.clone()
                        in_weights[i] = fields
                        weights.append(in_weights)
                    self.weights = torch.concat(weights, 1).permute(1, 0, 2, 3)
                self.weights.requires_grad_(True)

    def forward(self, x: torch.Tensor):
        self._update_weights()  # Update weights if necessary
        return torch.nn.functional.conv2d(x, self.weights, **self.kwargs)


class SampledSpatialReceptiveField2d(torch.nn.Module):
    """
    Creates a spatial receptive field as 2-dimensional convolutions, sampled over a set of scales,
    angles, ratios, and derivatives.
    This module allows for the optimization of the input parameters for scales, angles, and ratios
    (not derivatives) and will update the parameters (and, by extension, the receptive fields) accordingly
    if the respective parameters are set to True.
    This module is a wrapper around the `SpatialReceptiveField2d` module and will forward the kwargs.

    Example:
        >>> import torch
        >>> from norse.torch import SampledSpatialReceptiveField2d
        >>> scales = torch.tensor([1.0, 2.0])
        >>> angles = torch.tensor([0.0, 1.0])
        >>> ratios = torch.tensor([0.5, 1.0])
        >>> derivatives = torch.tensor([[0, 0]])
        >>> m = SampledSpatialReceptiveField2d(1, 9, scales, angles, ratios, derivatives,
        >>>                                    optimize_scales=False, optimize_angles=False,
        >>>                                    optimize_ratios=True)
        >>> optim = torch.optim.SGD(list(m.parameters()), lr=1)
        >>> y = m(torch.ones(1, 1, 9, 9))
        >>> y.sum().backward()
        >>> optim.step() # Will update the ratios
        >>> m.ratios() # Are now _different_ than the initial ratios
    """

    def __init__(
        self,
        in_channels: int,
        size: int,
        scales: torch.Tensor,
        angles: torch.Tensor,
        ratios: torch.Tensor,
        derivatives: torch.Tensor,
        optimize_scales: bool = True,
        optimize_angles: bool = True,
        optimize_ratios: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.scales = torch.nn.Parameter(scales) if optimize_scales else scales
        self.angles = torch.nn.Parameter(angles) if optimize_angles else angles
        self.ratios = torch.nn.Parameter(ratios) if optimize_ratios else ratios

        self.derivatives = derivatives
        self.has_updated = False

        self.submodule = SpatialReceptiveField2d(
            in_channels=in_channels,
            size=size,
            rf_parameters=spatial_parameters(
                self.scales, self.angles, self.ratios, self.derivatives
            ),
            optimize_fields=False,
            **kwargs,
        )

        if optimize_angles or optimize_scales or optimize_ratios:

            def update_hook(m, gi, go):
                self.has_updated = True

            self.register_full_backward_hook(update_hook)

    def forward(self, x: torch.Tensor):
        self._update_weights()
        return self.submodule(x)

    def _update_weights(self):
        if self.has_updated:
            self.submodule.rf_parameters = spatial_parameters(
                self.scales, self.angles, self.ratios, self.derivatives
            )
            self.submodule.has_updated = True
            self.submodule._update_weights()


class ParameterizedSpatialReceptiveField2d(torch.nn.Module):
    """
    A parameterized version of the `SpatialReceptiveField2d` module, where the scales, angles,
    and ratios are optimized and updated for each kernel individually during training.
    This is opposite to the `SampledSpatialReceptiveField2d` module, where the scales, angles,
    and ratios are updated individually (as generating functions for the kernels).
    This module wraps the `SpatialReceptiveField2d` module.
    This module is a wrapper around the `SpatialReceptiveField2d` module and will forward the kwargs.

    Example:
        >>> import torch
        >>> from norse.torch import ParameterizedSpatialReceptiveField2d
        >>> scales = torch.tensor([1.0, 2.0])
        >>> angles = torch.tensor([0.0, 1.0])
        >>> ratios = torch.tensor([0.5, 1.0])
        >>> m = ParameterizedSpatialReceptiveField2d(1, 9, scales, angles, ratios, 1,
        >>>                                          optimize_scales=False, optimize_angles=False,
        >>>                                          optimize_ratios=True)

    """

    def __init__(
        self,
        in_channels: int,
        size: int,
        scales: torch.Tensor,
        angles: torch.Tensor,
        ratios: torch.Tensor,
        derivatives: torch.Tensor,
        optimize_scales: bool = True,
        optimize_angles: bool = True,
        optimize_ratios: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.initial_parameters = spatial_parameters(
            scales, angles, ratios, derivatives
        )
        self.scales = (
            torch.nn.Parameter(self.initial_parameters[:, 0])
            if optimize_scales
            else self.initial_parameters[:, 0]
        )
        self.angles = (
            torch.nn.Parameter(self.initial_parameters[:, 1])
            if optimize_angles
            else self.initial_parameters[:, 1]
        )
        self.ratios = (
            torch.nn.Parameter(self.initial_parameters[:, 2])
            if optimize_ratios
            else self.initial_parameters[:, 2]
        )
        rf_parameters = torch.concat(
            [
                torch.stack([self.scales, self.angles, self.ratios], 1),
                self.initial_parameters[:, 3:],
            ],
            1,
        )
        self.submodule = SpatialReceptiveField2d(
            in_channels=in_channels,
            size=size,
            rf_parameters=rf_parameters,
            optimize_fields=False,
            **kwargs,
        )
        self.has_updated = False

        if optimize_angles or optimize_scales or optimize_ratios:

            def update_hook(m, gi, go):
                self.has_updated = True

            self.register_full_backward_hook(update_hook)

    def forward(self, x: torch.Tensor):
        self._update_weights()
        return self.submodule(x)

    def _update_weights(self):
        if self.has_updated:
            self.submodule.rf_parameters = torch.concat(
                [
                    torch.stack([self.scales, self.angles, self.ratios], 1),
                    self.initial_parameters[:, 3:],
                ],
                1,
            )
            self.submodule.has_updated = True
            self.submodule._update_weights()


class TemporalReceptiveField(torch.nn.Module):
    """Creates ``n_scales`` temporal receptive fields for arbitrary n-dimensional inputs.
    The scale spaces are selected in a range of [min_scale, max_scale] using an exponential distribution, scattered using ``torch.linspace``.

    Parameters:
        shape (torch.Size): The shape of the incoming tensor, where the first dimension denote channels
        n_scales (int): The number of temporal scale spaces to iterate over.
        activation (SNNCell): The activation neuron. Defaults to LIBoxCell
        activation_state_map (Callable): A function that takes a tensor and provides a neuron parameter tuple.
            Required if activation is changed, since the default behaviour provides LIBoxParameters.
        min_scale (float): The minimum scale space. Defaults to 1.
        max_scale (Optional[float]): The maximum scale. Defaults to None. If set, c is ignored.
        c (Optional[float]): The base from which to generate scale values. Should be a value
            between 1 to 2, exclusive. Defaults to sqrt(2). Ignored if max_scale is set.
        time_constants (Optional[torch.Tensor]): Hardcoded time constants. Will overwrite the automatically generated, logarithmically distributed scales, if set. Defaults to None.
        dt (float): Neuron simulation timestep. Defaults to 0.001.
    """

    def __init__(
        self,
        shape: torch.Size,
        n_scales: int = 4,
        activation: SNNCell.type = LIBoxCell,
        activation_state_map: Callable[
            [torch.Tensor], NamedTuple
        ] = lambda t: LIBoxParameters(tau_mem_inv=t),
        min_scale: float = 1,
        max_scale: Optional[float] = None,
        c: float = 1.41421,
        time_constants: Optional[torch.Tensor] = None,
        dt: float = 0.001,
    ):
        super().__init__()
        if time_constants is None:
            taus = (1 / dt) / temporal_scale_distribution(
                n_scales, min_scale=min_scale, max_scale=max_scale, c=c
            )
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
