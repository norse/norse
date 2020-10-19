import torch
import torch.jit
import numpy as np
from typing import Optional, Tuple

from ..functional.leaky_integrator import (
    li_step,
    li_feed_forward_step,
    LIState,
    LIParameters,
)


class LICell(torch.nn.Module):
    r"""Cell for a leaky-integrator.
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
        output_size: int,
        p: LIParameters = LIParameters(),
        dt: float = 0.001,
    ):
        super(LICell, self).__init__()
        self.input_weights = torch.nn.Parameter(
            torch.randn(output_size, input_size) / np.sqrt(input_size)
        )
        self.p = p
        self.dt = dt
        self.output_size = output_size

    def forward(
        self, input_tensor: torch.Tensor, state: Optional[LIState] = None
    ) -> Tuple[torch.Tensor, LIState]:
        if state is None:
            state = LIState(
                v=self.p.v_leak.detach(),
                i=torch.zeros(
                    (input_tensor.shape[0], self.output_size),
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


class LIFeedForwardCell(torch.nn.Module):
    r"""Cell for a leaky-integrator.
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

    def __init__(self, p: LIParameters = LIParameters(), dt: float = 0.001):
        super(LIFeedForwardCell, self).__init__()
        self.p = p
        self.dt = dt

    def forward(
        self, input_tensor: torch.Tensor, state: Optional[LIState] = None
    ) -> Tuple[torch.Tensor, LIState]:
        if state is None:
            state = LIState(
                v=self.p.v_leak.detach(),
                i=torch.zeros(
                    *input_tensor.shape,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
            )
            state.v.requires_grad = True
        return li_feed_forward_step(input_tensor, state, p=self.p, dt=self.dt)
