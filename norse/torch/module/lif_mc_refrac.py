from typing import Optional, Tuple

import numpy as np
import torch

from ..functional.lif_refrac import LIFRefracParameters, LIFRefracState
from ..functional.lif import LIFState
from ..functional.lif_mc_refrac import (
    lif_mc_refrac_step,
)

from norse.torch.module.snn import SNNRecurrentCell


class LIFMCRefracRecurrentCell(SNNRecurrentCell):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: LIFRefracParameters = LIFRefracParameters(),
        g_coupling: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # pytype: disable=wrong-arg-types
        super().__init__(
            activation=None,
            state_fallback=self.initial_state,
            input_size=input_size,
            hidden_size=hidden_size,
            p=p,
            **kwargs
        )
        # pytype: enable=wrong-arg-types

        self.g_coupling = (
            g_coupling
            if g_coupling is not None
            else torch.nn.Parameter(
                torch.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
            )
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIFRefracState:
        state = LIFRefracState(
            lif=LIFState(
                z=torch.zeros(
                    input_tensor.shape[0],
                    self.hidden_size,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                v=self.p.lif.v_leak.detach()
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
            ),
            rho=torch.zeros(
                input_tensor.shape[0],
                self.hidden_size,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
        )
        state.lif.v.requires_grad = True
        return state

    def forward(
        self, input_tensor: torch.Tensor, state: Optional[LIFRefracState] = None
    ) -> Tuple[torch.Tensor, LIFRefracState]:
        if state is None:
            state = self.initial_state(input_tensor)
        return lif_mc_refrac_step(
            input_tensor,
            state,
            self.input_weights,
            self.recurrent_weights,
            self.g_coupling,
            p=self.p,
            dt=self.dt,
        )
