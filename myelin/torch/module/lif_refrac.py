import torch

from ..functional.lif import LIFState, LIFFeedForwardState, LIFParameters

from ..functional.lif_refrac import (
    LIFRefracParameters,
    LIFRefracState,
    LIFRefracFeedForwardState,
    lif_refrac_step,
    lif_refrac_feed_forward_step,
)

from typing import Tuple
import numpy as np


class LIFRefracCell(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        p: LIFRefracParameters = LIFRefracParameters(),
        dt: float = 0.001,
    ):
        super(LIFRefracCell, self).__init__()
        self.input_weights = torch.nn.Parameter(
            torch.randn(hidden_size, input_size) / np.sqrt(input_size)
        )
        self.recurrent_weights = torch.nn.Parameter(
            torch.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        )
        self.hidden_size = hidden_size
        self.p = p
        self.dt = dt

    def initial_state(self, batch_size, device, dtype=torch.float) -> LIFRefracState:
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
        return lif_refrac_step(
            input,
            state,
            self.input_weights,
            self.recurrent_weights,
            p=self.p,
            dt=self.dt,
        )


class LIFRefracFeedForwardCell(torch.nn.Module):
    def __init__(
        self, shape, p: LIFRefracParameters = LIFRefracParameters(), dt: float = 0.001
    ):
        super(LIFRefracFeedForwardCell, self).__init__()
        self.shape = shape
        self.p = p
        self.dt = dt

    def initial_state(self, batch_size, device, dtype) -> LIFFeedForwardState:
        return LIFRefracFeedForwardState(
            LIFFeedForwardState(
                v=torch.zeros(batch_size, *self.shape, device=device, dtype=dtype),
                i=torch.zeros(batch_size, *self.shape, device=device, dtype=dtype),
            ),
            rho=torch.zeros(batch_size, *self.shape, device=device, dtype=dtype),
        )

    def forward(
        self, input: torch.Tensor, state: LIFRefracFeedForwardState
    ) -> Tuple[torch.Tensor, LIFRefracFeedForwardState]:
        return lif_refrac_feed_forward_step(input, state, p=self.p, dt=self.dt)
