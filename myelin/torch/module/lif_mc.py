import torch

from ..functional.lif import LIFState, LIFParameters
from ..functional.lif_mc import lif_mc_step

import numpy as np
from typing import Tuple


class LIFMCCell(torch.nn.Module):
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
