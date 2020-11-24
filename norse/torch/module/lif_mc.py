import torch

import numpy as np
from typing import Optional, Tuple

from ..functional.lif import LIFState, LIFParameters
from ..functional.lif_mc import lif_mc_step


class LIFMCCell(torch.nn.Module):
    r"""Computes a single euler-integration step of a LIF multi-compartment
    neuron-model.

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} \
            - g_{\text{coupling}} v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}})

    and transition equations

    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\text{reset}} \\
            i &= i + w_{\text{input}} z_{\text{in}} \\
            i &= i + w_{\text{rec}} z_{\text{rec}}
        \end{align*}

    where :math:`z_{\text{rec}}` and :math:`z_{\text{in}}` are the
    recurrent and input spikes respectively.


    Parameters:
        input_size (int): Size of the input. Also known as the number of input features.
        hidden_size (int): Size of the hidden state. Also known as the number of input features.
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
        super(LIFMCCell, self).__init__()

        self.input_weights = torch.nn.Parameter(
            torch.randn(hidden_size, input_size) / np.sqrt(input_size)
        )
        self.recurrent_weights = torch.nn.Parameter(
            torch.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        )
        self.g_coupling = torch.nn.Parameter(
            torch.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        )
        self.hidden_size = hidden_size
        self.p = p
        self.dt = dt

    def forward(
        self, input_tensor: torch.Tensor, state: Optional[LIFState] = None
    ) -> Tuple[torch.Tensor, LIFState]:
        if state is None:
            state = LIFState(
                z=torch.zeros(
                    input_tensor.shape[0],
                    self.hidden_size,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                v=self.p.v_leak.detach()
                * torch.ones(
                    input_tensor.shape[0],
                    self.hidden_size,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                i=torch.zeros(
                    input_tensor.shape[0],
                    self.hidden_size,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
            )
        return lif_mc_step(
            input_tensor,
            state,
            self.input_weights,
            self.recurrent_weights,
            self.g_coupling,
            p=self.p,
            dt=self.dt,
        )
