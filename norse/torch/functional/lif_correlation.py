import torch
import torch.jit

from .lif import LIFState, LIFParameters, lif_step

from .correlation_sensor import (
    CorrelationSensorState,
    CorrelationSensorParameters,
    correlation_sensor_step,
)

from typing import NamedTuple, Tuple


class LIFCorrelationState(NamedTuple):
    lif_state: LIFState
    input_correlation_state: CorrelationSensorState
    recurrent_correlation_state: CorrelationSensorState


class LIFCorrelationParameters(NamedTuple):
    lif_parameters: LIFParameters = LIFParameters()
    input_correlation_parameters: CorrelationSensorParameters = CorrelationSensorParameters()
    recurrent_correlation_parameters: CorrelationSensorParameters = CorrelationSensorParameters()


def lif_correlation_step(
    input_tensor: torch.Tensor,
    state: LIFCorrelationState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    parameters: LIFCorrelationParameters = LIFCorrelationParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFCorrelationState]:
    z_new, s_new = lif_step(
        input_tensor, state.lif_state, input_weights, recurrent_weights, parameters.lif_parameters, dt
    )

    input_correlation_state_new = correlation_sensor_step(
        z_pre=input_tensor,
        z_post=z_new,
        state=state.input_correlation_state,
        parameters=parameters.input_correlation_parameters,
        dt=dt,
    )

    recurrent_correlation_state_new = correlation_sensor_step(
        z_pre=state.lif_state.z,
        z_post=z_new,
        state=state.recurrent_correlation_state,
        parameters=parameters.recurrent_correlation_parameters,
        dt=dt,
    )
    return (
        z_new,
        LIFCorrelationState(
            lif_state=s_new,
            input_correlation_state=input_correlation_state_new,
            recurrent_correlation_state=recurrent_correlation_state_new,
        ),
    )
