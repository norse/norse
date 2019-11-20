import torch

from ..functional.coba_lif import CobaLIFParameters, CobaLIFState, coba_lif_step

from typing import Tuple
import numpy as np


class CobaLIFCell(torch.nn.Module):
    """Module that computes a single euler-integration step of a conductance based
    LIF neuron-model. More specifically it implements one integration step of the following ODE

    .. math::
        \\begin{align*}
            \dot{v} &= 1/c_{\\text{mem}} (g_l (v_{\\text{leak}} - v) + g_e (E_{\\text{rev_e}} - v) + g_i (E_{\\text{rev_i}} - v)) \\\\
            \dot{g_e} &= -1/\\tau_{\\text{syn}} g_e \\\\
            \dot{g_i} &= -1/\\tau_{\\text{syn}} g_i
        \end{align*}

    together with the jump condition
    
    .. math::
        z = \Theta(v - v_{\\text{th}})
    
    and transition equations

    .. math::
        \\begin{align*}
            v &= (1-z) v + z v_{\\text{reset}} \\\\
            g_e &= g_e + \\text{relu}(w_{\\text{input}}) z_{\\text{in}} \\\\
            g_e &= g_e + \\text{relu}(w_{\\text{rec}}) z_{\\text{rec}} \\\\
            g_i &= g_i + \\text{relu}(-w_{\\text{input}}) z_{\\text{in}} \\\\
            g_i &= g_i + \\text{relu}(-w_{\\text{rec}}) z_{\\text{rec}} \\\\
        \end{align*}

    where :math:`z_{\\text{rec}}` and :math:`z_{\\text{in}}` are the recurrent and input
    spikes respectively.

    Parameters:
        input_size (int): Size of the input.
        hidden_size (int): Size of the hidden state.
        p (LIFParameters): Parameters of the LIF neuron model.
        dt (float): Time step to use.

    Examples:

        >>> batch_size = 16
        >>> lif = CobaLIFCell(10, 20)
        >>> input = torch.randn(batch_size, 10)
        >>> s0 = lif.initial_state(batch_size)
        >>> output, s0 = lif(input, s0)
    """
    def __init__(
        self,
        input_size,
        hidden_size,
        p: CobaLIFParameters = CobaLIFParameters(),
        dt: float = 0.001,
    ):
        super(CobaLIFCell, self).__init__()
        self.input_weights = torch.nn.Parameter(
            torch.randn(hidden_size, input_size) / np.sqrt(input_size)
        )
        self.recurrent_weights = torch.nn.Parameter(
            torch.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        )
        self.p = p
        self.dt = dt

    def initial_state(
        self, batch_size: int, device: torch.device, dtype=torch.float
    ) -> CobaLIFState:
        return CobaLIFState(
            z=torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
            v=torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
            g_e=torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
            g_i=torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
        )

    def forward(
        self, input: torch.Tensor, state: CobaLIFState
    ) -> Tuple[torch.Tensor, CobaLIFState]:
        return coba_lif_step(
            input,
            state,
            self.input_weights,
            self.recurrent_weights,
            p=self.p,
            dt=self.dt,
        )
