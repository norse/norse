import torch

from typing import Optional, Tuple

from norse.torch.functional.lif_correlation import (
    LIFCorrelationState,
    LIFCorrelationParameters,
    lif_correlation_step,
)
from norse.torch.functional.lif import LIFState
from norse.torch.functional.correlation_sensor import CorrelationSensorState


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

    def forward(
        self,
        input_tensor: torch.Tensor,
        input_weights: torch.Tensor,
        recurrent_weights: torch.Tensor,
        state: Optional[LIFCorrelationState],
    ) -> Tuple[torch.Tensor, LIFCorrelationState]:
        if state is None:
            hidden_features = self.hidden_size
            input_features = self.input_size
            batch_size = input_tensor.shape[0]
            state = LIFCorrelationState(
                lif_state=LIFState(
                    z=torch.zeros(
                        batch_size,
                        hidden_features,
                        device=input_tensor.device,
                        dtype=input_tensor.dtype,
                    ),
                    v=self.p.lif_parameters.v_leak.detach(),
                    i=torch.zeros(
                        batch_size,
                        hidden_features,
                        device=input_tensor.device,
                        dtype=input_tensor.dtype,
                    ),
                ),
                input_correlation_state=CorrelationSensorState(
                    post_pre=torch.zeros(
                        (batch_size, input_features, hidden_features),
                        device=input_tensor.device,
                        dtype=input_tensor.dtype,
                    ),
                    correlation_trace=torch.zeros(
                        (batch_size, input_features, hidden_features),
                        device=input_tensor.device,
                        dtype=input_tensor.dtype,
                    ).float(),
                    anti_correlation_trace=torch.zeros(
                        (batch_size, input_features, hidden_features),
                        device=input_tensor.device,
                        dtype=input_tensor.dtype,
                    ).float(),
                ),
                recurrent_correlation_state=CorrelationSensorState(
                    correlation_trace=torch.zeros(
                        (batch_size, hidden_features, hidden_features),
                        device=input_tensor.device,
                        dtype=input_tensor.dtype,
                    ),
                    anti_correlation_trace=torch.zeros(
                        (batch_size, hidden_features, hidden_features),
                        device=input_tensor.device,
                        dtype=input_tensor.dtype,
                    ),
                    post_pre=torch.zeros(
                        (batch_size, hidden_features, hidden_features),
                        device=input_tensor.device,
                        dtype=input_tensor.dtype,
                    ),
                ),
            )
        return lif_correlation_step(
            input_tensor,
            state,
            input_weights,
            recurrent_weights,
            self.p,
            self.dt,
        )
