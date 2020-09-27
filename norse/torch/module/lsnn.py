from typing import Optional, Tuple

import torch
import numpy as np

from ..functional.lsnn import (
    LSNNParameters,
    LSNNState,
    LSNNFeedForwardState,
    lsnn_step,
    lsnn_feed_forward_step,
)


class LSNNCell(torch.nn.Module):
    r"""Module that computes a single euler-integration step of a LSNN
    neuron-model. More specifically it implements one integration step of
    the following ODE

    .. math::
        \\begin{align*}
            \dot{v} &= 1/\\tau_{\\text{mem}} (v_{\\text{leak}} - v + i) \\\\
            \dot{i} &= -1/\\tau_{\\text{syn}} i \\\\
            \dot{b} &= -1/\\tau_{b} b
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\\text{th}} + b)

    and transition equations

    .. math::
        \\begin{align*}
            v &= (1-z) v + z v_{\\text{reset}} \\\\
            i &= i + w_{\\text{input}} z_{\\text{in}} \\\\
            i &= i + w_{\\text{rec}} z_{\\text{rec}} \\\\
            b &= b + \\beta z
        \end{align*}

    where :math:`z_{\\text{rec}}` and :math:`z_{\\text{in}}` are the
    recurrent and input spikes respectively.

    Parameters:
        input (torch.Tensor): the input spikes at the current time step
        s (LSNNState): current state of the lsnn unit
        input_weights (torch.Tensor): synaptic weights for input spikes
        recurrent_weights (torch.Tensor): synaptic weights for recurrent spikes
        p (LSNNParameters): parameters of the lsnn unit
        dt (float): Integration timestep to use
    """

    def __init__(
        self,
        input_features,
        output_features,
        p: LSNNParameters = LSNNParameters(),
        dt: float = 0.001,
    ):
        super(LSNNCell, self).__init__()
        self.input_weights = torch.nn.Parameter(
            torch.randn(output_features, input_features) / np.sqrt(input_features)
        )
        self.recurrent_weights = torch.nn.Parameter(
            torch.randn(output_features, output_features)
        )
        self.input_features = input_features
        self.output_features = output_features
        self.p = p
        self.dt = dt

    def forward(
        self, input_tensor: torch.Tensor, state: Optional[LSNNState] = None
    ) -> Tuple[torch.Tensor, LSNNState]:
        if state is None:
            state = LSNNState(
                z=torch.zeros(
                    input_tensor.shape[0],
                    self.output_features,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                v=self.p.v_leak,
                i=torch.zeros(
                    input_tensor.shape[0],
                    self.output_features,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                b=torch.zeros(
                    input_tensor.shape[0],
                    self.output_features,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
            )
        return lsnn_step(
            input_tensor,
            state,
            self.input_weights,
            self.recurrent_weights,
            p=self.p,
            dt=self.dt,
        )


class LSNNLayer(torch.nn.Module):
    r"""A Long short-term memory neuron module adapted from
        https://arxiv.org/abs/1803.09574

    Usage:
      >>> from norse.torch.module import LSNNLayer, LSNNCell
      >>> layer = LSNNLayer(LSNNCell, 2, 10)    // LSNNCell of shape 2 -> 10
      >>> data  = torch.zeros(2, 5, 2)          // Arbitrary data
      >>> output, state = layer.forward(data)

    Parameters:
      cell (torch.nn.Module): the underling neuron module, uninitialized
      *cell_args: variable length input arguments for the underlying cell
                  constructor
      **cell_kwargs: variable length key-value arguments for the underlying cell constructor
    """

    def __init__(self, cell, *cell_args, **cell_kwargs):
        super(LSNNLayer, self).__init__()
        self.cell = cell(*cell_args, **cell_kwargs)

    def forward(
        self, input_tensor: torch.Tensor, state: Optional[LSNNState] = None
    ) -> Tuple[torch.Tensor, LSNNState]:
        """
        Takes one step in the LSNN layer by simulating the layer for a number of timesteps.
        Since the layer is recurrent, each simulation state (LSNNState) is used as the input for the next step.

        The function expects inputs in the shape (simulation time steps, batch size, ...).

        Parameters:
        input_tensor (torch.Tensor): Input tensor with timesteps in the first dimension
        state (Optional[LSNNState]): The input LSNN state. Defaults to None on the first timestep

        Returns:
        A tuple of 1) spikes from each timestep and 2) the LSNNState from the last timestep.
        """
        inputs = input_tensor.unbind(0)
        outputs = []  # torch.jit.annotate(List[torch.Tensor], [])
        for input_step in inputs:
            out, state = self.cell(input_step, state)
            outputs += [out]
        return torch.stack(outputs), state


class LSNNFeedForwardCell(torch.nn.Module):
    r"""Euler integration cell for LIF Neuron with threshold adaptation.
    More specifically it implements one integration step of the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\\text{syn}} i \\
            \dot{b} &= -1/\tau_{b} b
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}} + b)

    and transition equations

    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\text{reset}} \\
            i &= i + \text{input} \\
            b &= b + \beta z
        \end{align*}

    Parameters:
        input (torch.Tensor): the input spikes at the current time step
        s (LSNNFeedForwardState): current state of the lsnn unit
        p (LSNNParameters): parameters of the lsnn unit
        dt (float): Integration timestep to use
    """

    def __init__(self, shape, p: LSNNParameters = LSNNParameters(), dt: float = 0.001):
        super(LSNNFeedForwardCell, self).__init__()
        self.shape = shape
        self.p = p
        self.dt = dt

    def forward(
        self, input_tensor: torch.Tensor, state: Optional[LSNNFeedForwardState] = None
    ) -> Tuple[torch.Tensor, LSNNFeedForwardState]:
        if state is None:
            state = LSNNFeedForwardState(
                v=self.p.v_leak,
                i=torch.zeros(
                    input_tensor.shape[0],
                    self.output_features,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                b=torch.zeros(
                    input_tensor.shape[0],
                    self.output_features,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
            )
        return lsnn_feed_forward_step(input_tensor, state, p=self.p, dt=self.dt)
