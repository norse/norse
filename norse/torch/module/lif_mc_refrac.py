import torch

from ..functional.lif_refrac import LIFRefracParameters, LIFRefracState
from ..functional.lif import LIFState
from ..functional.lif_mc_refrac import lif_mc_refrac_step

import numpy as np
from typing import Optional, Tuple


class LIFMCRefracCell(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: LIFRefracParameters = LIFRefracParameters(),
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

    def forward(
        self, input_tensor: torch.Tensor, state: Optional[LIFRefracState] = None
    ) -> Tuple[torch.Tensor, LIFRefracState]:
        if state is None:
            state = LIFRefracState(
                lif=LIFState(
                    z=torch.zeros(
                        input_tensor.shape[0],
                        self.hidden_size,
                        device=input_tensor.device,
                        dtype=input_tensor.dtype,
                    ),
                    v=self.p.lif.v_leak,
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
        return lif_mc_refrac_step(
            input_tensor,
            state,
            self.input_weights,
            self.recurrent_weights,
            self.g_coupling,
            p=self.p,
            dt=self.dt,
        )
