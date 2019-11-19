from typing import Tuple, List

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
    """Module that computes a single euler-integration step of a LSNN neuron-model.
    More specifically it implements one integration step of the following ODE

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

    where :math:`z_{\\text{rec}}` and :math:`z_{\\text{in}}` are the recurrent and input
    spikes respectively.

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

    def initial_state(self, batch_size, device, dtype=torch.float) -> LSNNState:
        """return the initial state of an LSNN neuron"""
        return LSNNState(
            z=torch.zeros(batch_size, self.output_features, device=device, dtype=dtype),
            v=torch.zeros(batch_size, self.output_features, device=device, dtype=dtype),
            i=torch.zeros(batch_size, self.output_features, device=device, dtype=dtype),
            b=torch.zeros(batch_size, self.output_features, device=device, dtype=dtype),
        )

    def forward(
        self, input: torch.Tensor, state: LSNNState
    ) -> Tuple[torch.Tensor, LSNNState]:
        return lsnn_step(
            input,
            state,
            self.input_weights,
            self.recurrent_weights,
            p=self.p,
            dt=self.dt,
        )


class LSNNLayer(torch.nn.Module):
    def __init__(self, cell, *cell_args):
        super(LSNNLayer, self).__init__()
        self.cell = cell(*cell_args)

    def initial_state(self, batch_size, device, dtype=torch.float) -> LSNNState:
        """Return the initial state of the LSNN layer, as given by the internal LSNNCell"""
        return self.cell.initial_state(batch_size, device, dtype)

    def forward(
        self, input: torch.Tensor, state: LSNNState
    ) -> Tuple[torch.Tensor, LSNNState]:
        inputs = input.unbind(0)
        outputs = []  # torch.jit.annotate(List[torch.Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class LSNNFeedForwardCell(torch.nn.Module):
    """Euler integration cell for LIF Neuron with threshhold adaptation.
    More specifically it implements one integration step of the following ODE

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
            i &= i + \\text{input} \\\\
            b &= b + \\beta z
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

    def initial_state(
            self, batch_size, device, dtype=torch.float
    ) -> LSNNFeedForwardState:
        """return the initial state of an LSNN neuron"""
        return LSNNFeedForwardState(
            v=torch.zeros(batch_size, *self.shape, device=device, dtype=dtype),
            i=torch.zeros(batch_size, *self.shape, device=device, dtype=dtype),
            b=torch.zeros(batch_size, *self.shape, device=device, dtype=dtype),
        )

    def forward(
        self, input: torch.Tensor, state: LSNNFeedForwardState
    ) -> Tuple[torch.Tensor, LSNNFeedForwardState]:
        return lsnn_feed_forward_step(input, state, p=self.p, dt=self.dt)
