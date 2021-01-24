"""
Leaky integrate-and-fire neurons is a popular neuron model
for spiking neural networks because they are simple and
fast while still being biologically interesting.
"""
from typing import Any, Optional, Tuple

import numpy as np
import torch

from norse.torch.functional.lif import (
    LIFState,
    LIFFeedForwardState,
    LIFParameters,
    lif_step,
    lif_feed_forward_step,
)
from norse.torch.module.snn import FeedforwardSNN, SNNCell, RecurrentSNN
from norse.torch.module.util import remove_autopses


class FeedforwardLIF(FeedforwardSNN):
    """
    A neuron layer that wraps a recurrent LIFCell in time such
    that the layer keeps track of temporal sequences of spikes.
    After application, the layer returns a tuple containing
      (spikes from all timesteps, state from the last timestep).

    Example:
        >>> data = torch.zeros(10, 5, 2) # 10 timesteps, 5 batches, 2 neurons
        >>> l = LIFFeedForwardLayer()
        >>> l(data) # Returns tuple of (Tensor(10, 5, 4), LIFState)

    Arguments:
        p (LIFParameters): Parameters of the LIF neuron model.
        dt (float): Time step to use. Defaults to 0.001.
    """

    def __init__(self, p: LIFParameters = LIFParameters(), dt: float = 0.001):
        super().__init__(lif_feed_forward_step, self.initial_state, p, dt)

    def initial_state(self, input_tensor: torch.Tensor) -> LIFFeedForwardState:
        state = LIFFeedForwardState(
            v=self.p.v_leak.detach(),
            i=torch.zeros(
                *input_tensor.shape,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
        )
        state.v.requires_grad = True
        return state


class RecurrentLIFCell(RecurrentSNNCell):
    """Module that computes a single euler-integration step of a
    LIF neuron-model with recurrence.
    More specifically it implements one integration step
    of the following ODE

    .. math::
        \\begin{align*}
            \\dot{v} &= 1/\\tau_{\\text{mem}} (v_{\\text{leak}} - v + i) \\
            \\dot{i} &= -1/\\tau_{\\text{syn}} i
        \\end{align*}

    together with the jump condition

    .. math::
        z = \\Theta(v - v_{\\text{th}})

    and transition equations

    .. math::
        \\begin{align*}
            v &= (1-z) v + z v_{\\text{reset}} \\
            i &= i + w_{\\text{input}} z_{\\text{in}} \\
            i &= i + w_{\\text{rec}} z_{\\text{rec}}
        \\end{align*}

    where :math:`z_{\\text{rec}}` and :math:`z_{\\text{in}}` are the
    recurrent and input spikes respectively.

    Parameters:
        input_size (int): Size of the input. Also known as the number of input features.
        hidden_size (int): Size of the hidden state. Also known as the number of input features.
        p (LIFParameters): Parameters of the LIF neuron model.
        dt (float): Time step to use.
        autopses (bool): Allow self-connections in the recurrence? Defaults to False.

    Examples:

        >>> batch_size = 16
        >>> lif = LIFCell(10, 20)
        >>> input = torch.randn(batch_size, 10)
        >>> output, s0 = lif(input)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: LIFParameters = LIFParameters(),
        dt: float = 0.001,
        autopses: bool = False,
    ):
        super().__init__(
            input_size,
            hidden_size,
            activation=lif_step,
            state_fallback=self.initial_state,
            p=p,
            dt=dt,
            autopses=autopses,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIFState:
        state = LIFState(
            z=torch.zeros(
                input_tensor.shape[0],
                self.hidden_size,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            v=self.p.v_leak.detach(),
            i=torch.zeros(
                input_tensor.shape[0],
                self.hidden_size,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
        )
        state.v.requires_grad = True
        return state


class RecurrentLIF(RecurrentSNN):
    """
    A neuron layer that wraps a recurrent LIFCell in time such
    that the layer keeps track of temporal sequences of spikes.
    After application, the layer returns a tuple containing
      (spikes from all timesteps, state from the last timestep).

    Example:
        >>> data = torch.zeros(10, 5, 2) # 10 timesteps, 5 batches, 2 neurons
        >>> l = LIFLayer(2, 4)
        >>> l(data) # Returns tuple of (Tensor(10, 5, 4), LIFState)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: LIFParameters = LIFParameters(),
        dt: float = 0.001,
        autopses: bool = False,
    ):
        super().__init__(
            input_size,
            hidden_size,
            activation=lif_step,
            state_fallback=self.initial_state,
            p=p,
            dt=dt,
            autopses=autopses,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIFState:
        state = LIFState(
            z=torch.zeros(
                input_tensor.shape[0],
                self.hidden_size,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            v=self.p.v_leak.detach(),
            i=torch.zeros(
                input_tensor.shape[0],
                self.hidden_size,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
        )
        state.v.requires_grad = True
        return state
