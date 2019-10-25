import torch

from ..functional.lif_correlation import (
    LIFCorrelationState,
    LIFCorrelationParameters,
    lif_correlation_step,
)
from ..functional.lif import LIFState
from ..functional.correlation_sensor import CorrelationSensorState

import numpy as np
from typing import Tuple


class LIFCorrelation(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        p: LIFCorrelationParameters = LIFCorrelationParameters(),
        dt: float = 0.001,
    ):
        super(LIFCorrelation, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.p = p
        self.dt = dt

    def initial_state(
        self, batch_size, device, dtype=torch.float
    ) -> LIFCorrelationState:
        hidden_features = self.hidden_size
        input_features = self.input_size

        return LIFCorrelationState(
            lif_state=LIFState(
                z=torch.zeros(batch_size, hidden_features, device=device, dtype=dtype),
                v=torch.zeros(batch_size, hidden_features, device=device, dtype=dtype),
                i=torch.zeros(batch_size, hidden_features, device=device, dtype=dtype),
            ),
            input_correlation_state=CorrelationSensorState(
                post_pre=torch.tensor(
                    np.zeros((batch_size, input_features, hidden_features)),
                    device=device,
                    dtype=dtype,
                ),
                correlation_trace=torch.tensor(
                    np.zeros((batch_size, input_features, hidden_features)),
                    device=device,
                    dtype=dtype,
                ).float(),
                anti_correlation_trace=torch.tensor(
                    np.zeros((batch_size, input_features, hidden_features)),
                    device=device,
                    dtype=dtype,
                ).float(),
            ),
            recurrent_correlation_state=CorrelationSensorState(
                correlation_trace=torch.tensor(
                    np.zeros((batch_size, hidden_features, hidden_features)),
                    device=device,
                    dtype=dtype,
                ),
                anti_correlation_trace=torch.tensor(
                    np.zeros((batch_size, hidden_features, hidden_features)),
                    device=device,
                    dtype=dtype,
                ),
                post_pre=torch.tensor(
                    np.zeros((batch_size, hidden_features, hidden_features)),
                    device=device,
                    dtype=dtype,
                ),
            ),
        )

    def forward(
        self,
        input: torch.Tensor,
        s: LIFCorrelationState,
        input_weights: torch.Tensor,
        recurrent_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, LIFCorrelationState]:
        return lif_correlation_step(
            input, s, input_weights, recurrent_weights, self.p, self.dt
        )
