r"""
Leaky integrators describe a *leaky* neuron membrane that integrates
incoming currents over time, but never spikes. In other words, the
neuron adds up incoming input current, while leaking out some of it
in every timestep.

See :mod:`norse.torch.functional.leaky_integrator` for more information.
"""
from typing import Optional, Tuple

import torch
import torch.jit
import numpy as np

from norse.torch.module.snn import SNN, SNNCell

from ..functional.leaky_integrator import (
    li_step,
    li_feed_forward_step,
    LIState,
    LIParameters,
)


class LICell(SNNCell):
    r"""Cell for a leaky-integrator *without* recurrence.
    More specifically it implements a discretized version of the ODE

    .. math::

        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}


    and transition equations

    .. math::
        i = i + w i_{\text{in}}

    Parameters:
        p (LIParameters): parameters of the leaky integrator
        dt (float): integration timestep to use
    """

    def __init__(self, p: LIParameters = LIParameters(), **kwargs):
        super().__init__(
            activation=li_feed_forward_step,
            state_fallback=self.initial_state,
            p=p,
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIState:
        state = LIState(
            v=self.p.v_leak.detach(),
            i=torch.zeros(
                *input_tensor.shape,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
        )
        state.v.requires_grad = True
        return state


class LI(SNN):
    r"""A neuron layer that wraps a leaky-integrator :class:`LICell` in time, but
    *without* recurrence. The layer iterates over the  _outer_ dimension of the input.
    More specifically it implements a discretized version of the ODE

    .. math::

        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}


    and transition equations

    .. math::
        i = i + w i_{\text{in}}

    After application, the layer returns a tuple containing
      (voltages from all timesteps, state from the last timestep).

    Example:
    >>> data = torch.zeros(10, 2) # 10 timesteps, 2 neurons
    >>> l = LI()
    >>> l(data) # Returns tuple of (Tensor(10, 2), LIState)

    Parameters:
        p (LIParameters): parameters of the leaky integrator
        dt (float): integration timestep to use
    """

    def __init__(self, p: LIParameters = LIParameters(), **kwargs):
        super().__init__(
            activation=li_feed_forward_step,
            state_fallback=self.initial_state,
            p=p,
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIState:
        state = LIState(
            v=torch.full(
                input_tensor.shape[1:],  # Assume first dimension is time
                self.p.v_leak.detach(),
                device=input_tensor.device,
                dtype=torch.float32,
            ),
            i=torch.zeros(
                *input_tensor.shape[1:],
                device=input_tensor.device,
                dtype=torch.float32,
            ),
        )
        state.v.requires_grad = True
        return state


class LILinearCell(torch.nn.Module):
    r"""Cell for a leaky-integrator with an additional linear weighting.
    More specifically it implements a discretized version of the ODE

    .. math::

        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}


    and transition equations

    .. math::
        i = i + w i_{\text{in}}

    Parameters:
        input_size (int): Size of the input. Also known as the number of input features.
        hidden_size (int): Size of the hidden state. Also known as the number of input features.
        p (LIParameters): parameters of the leaky integrator
        dt (float): integration timestep to use
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: LIParameters = LIParameters(),
        dt: float = 0.001,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.p = p
        self.dt = dt
        self.input_weights = torch.nn.Parameter(
            torch.randn(hidden_size, input_size) / np.sqrt(input_size)
        )

    def forward(
        self, input_tensor: torch.Tensor, state: Optional[LIState] = None
    ) -> Tuple[torch.Tensor, LIState]:
        if state is None:
            state = LIState(
                v=self.p.v_leak.detach(),
                i=torch.zeros(
                    (input_tensor.shape[0], self.hidden_size),
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
            )
            state.v.requires_grad = True
        return li_step(
            input_tensor,
            state,
            self.input_weights,
            p=self.p,
            dt=self.dt,
        )
