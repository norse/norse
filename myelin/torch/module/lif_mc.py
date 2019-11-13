import torch

from ..functional.lif import LIFState, LIFParameters
from ..functional.lif_mc import lif_mc_step

import numpy as np
from typing import Tuple


class LIFMCCell(torch.nn.Module):
    """Computes a single euler-integration step of a LIF multi-compartment
    neuron-model.

    .. math::
        \\begin{align*}
            \\dot{v} &= 1/\\tau_{\\text{mem}} (v_{\\text{leak}} - g_{\\text{coupling}} v + i) \\\\
            \\dot{i} &= -1/\\tau_{\\text{syn}} i
        \end{align*}

    together with the jump condition
    
    .. math::
        z = \Theta(v - v_{\\text{th}})
    
    and transition equations

    .. math::
        \\begin{align*}
            v &= (1-z) v + z v_{\\text{reset}} \\\\
            i &= i + w_{\\text{input}} z_{\\text{in}} \\\\
            i &= i + w_{\\text{rec}} z_{\\text{rec}}
        \end{align*}

    where :math:`z_{\\text{rec}}` and :math:`z_{\\text{in}}` are the recurrent and input
    spikes respectively.
    

    Parameters:
        input (torch.Tensor): the input spikes at the current time step
        s (LIFState): current state of the neuron
        input_weights (torch.Tensor): synaptic weights for incoming spikes
        recurrent_weights (torch.Tensor): synaptic weights for recurrent spikes
        g_coupling (torch.Tensor): conductances between the neuron compartments
        p (LIFParameters): neuron parameters
        dt (float): Integration timestep to use
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: LIFParameters = LIFParameters(),
        dt: float = 0.001,
    ):
        self.input_weights = torch.nn.Parameter(
            torch.randn(hidden_size, input_size) / np.sqrt(input_size)
        )
        self.recurrent_weights = torch.nn.Parameter(
            torch.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        )
        self.g_coupling = torch.nn.Parameter(
            torch.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        )
        self.p = p
        self.dt = dt

    def initial_state(
        self, batch_size: int, device: torch.device, dtype=torch.float
    ) -> LIFState:
        return LIFState(
            z=torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
            v=torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
            i=torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
        )

    def forward(
        self, input: torch.Tensor, state: LIFState
    ) -> Tuple[torch.Tensor, LIFState]:
        return lif_mc_step(
            input,
            state,
            self.input_weights,
            self.recurrent_weights,
            self.g_coupling,
            p=self.p,
            dt=self.dt,
        )
