import torch

from ..functional.lif_refrac import LIFRefracParameters, LIFRefracState
from ..functional.lif import LIFState
from ..functional.lif_mc_refrac import lif_mc_refrac_step

import numpy as np
from typing import Tuple


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

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.float = torch.float
    ) -> LIFRefracState:
        return LIFRefracState(
            lif=LIFState(
                z=torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
                v=torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
                i=torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
            ),
            rho=torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype),
        )

    def forward(
        self, input: torch.Tensor, state: LIFRefracState
    ) -> Tuple[torch.Tensor, LIFRefracState]:
        return lif_mc_refrac_step(
            input,
            state,
            self.input_weights,
            self.recurrent_weights,
            self.g_coupling,
            p=self.p,
            dt=self.dt,
        )
