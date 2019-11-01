import torch

from ..functional.coba_lif import CobaLIFParameters, CobaLIFState, coba_lif_step

from typing import Tuple
import numpy as np


class CobaLIFCell(torch.nn.Module):
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
