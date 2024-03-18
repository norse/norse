from typing import Optional, Tuple

import numpy as np
import torch

from norse.torch.functional.lif import LIFState, LIFParameters
from norse.torch.functional.lif_mc import lif_mc_step

from norse.torch.module.snn import SNNRecurrentCell


class LIFMCRecurrentCell(SNNRecurrentCell):
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
        autapses (bool): Allow self-connections in the recurrence? Defaults to False.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: LIFParameters = LIFParameters(),
        g_coupling: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # pytype: disable=wrong-arg-types
        super().__init__(
            activation=None,
            state_fallback=self.initial_state,
            input_size=input_size,
            hidden_size=hidden_size,
            p=p,
            **kwargs,
        )
        # pytype: enable=wrong-arg-types
        self.g_coupling = (
            g_coupling
            if g_coupling is not None
            else torch.nn.Parameter(
                torch.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
            )
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIFState:
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
        state.v.requires_grad = True
        return state

    def forward(
        self, input_tensor: torch.Tensor, state: Optional[LIFState] = None
    ) -> Tuple[torch.Tensor, LIFState]:
        if state is None:
            state = self.initial_state(input_tensor)
        return lif_mc_step(
            input_tensor,
            state,
            self.input_weights,
            self.recurrent_weights,
            self.g_coupling,
            p=self.p,
            dt=self.dt,
        )
