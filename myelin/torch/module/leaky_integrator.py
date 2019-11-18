import torch
import torch.jit
import numpy as np

from ..functional.leaky_integrator import (
    li_step,
    li_feed_forward_step,
    LIState,
    LIParameters,
)

from typing import Tuple


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
        input_features (int); Input feature dimension
        output_features (int): Output feature dimension
        p (LIParameters): parameters of the leaky integrator
        dt (float): integration timestep to use
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        p: LIParameters = LIParameters(),
        dt: float = 0.001,
    ):
        super(LICell, self).__init__()
        self.input_weights = torch.nn.Parameter(
            torch.randn(output_features, input_features) / np.sqrt(input_features)
        )
        self.p = p
        self.dt = dt
        self.output_features = output_features

    def initial_state(self, batch_size, device, dtype=torch.float) -> LIState:
        return LIState(
            v=torch.zeros(self.output_features, device=device, dtype=dtype),
            i=torch.zeros(self.output_features, device=device, dtype=dtype),
        )

    def forward(self, input: torch.Tensor, s: LIState) -> Tuple[torch.Tensor, LIState]:
        return li_step(input, s, self.input_weights, p=self.p, dt=self.dt)


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
        shape: Shape of the preprocessed input spikes
        p (LIParameters): parameters of the leaky integrator
        dt (float): integration timestep to use
    """
    def __init__(self, shape, p: LIParameters = LIParameters(), dt: float = 0.001):
        super(LIFeedForwardCell, self).__init__()
        self.p = p
        self.dt = dt
        self.shape = shape

    def initial_state(self, batch_size, device, dtype=torch.float):
        return LIState(
            v=torch.zeros(batch_size, *self.shape, device=device, dtype=dtype),
            i=torch.zeros(batch_size, *self.shape, device=device, dtype=dtype),
        )

    def forward(self, input: torch.Tensor, s: LIState) -> Tuple[torch.Tensor, LIState]:
        return li_feed_forward_step(input, s, p=self.p, dt=self.dt)
