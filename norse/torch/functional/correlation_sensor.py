import torch
import torch.jit

from typing import NamedTuple

from .heaviside import heaviside


@torch.jit.script
def pre_mask(weights, z):
    """Computes the mask produced by the pre-synaptic spikes on
    the synapse array."""
    return torch.transpose(torch.transpose(torch.zeros_like(weights), 1, 2) + z, 1, 2)


@torch.jit.script
def post_mask(weights, z):
    """Computes the mask produced by post-synaptic spikes on
    the synapse array.
    """
    return torch.zeros_like(weights) + z


@torch.jit.script
def post_pre_update(post_pre, post_spike_mask, pre_spike_mask):
    """Computes which synapses in the synapse array should be updated."""
    return heaviside(post_pre + post_spike_mask - pre_spike_mask)


class CorrelationSensorParameters(NamedTuple):
    eta_p: torch.Tensor = torch.as_tensor(1.0)
    eta_m: torch.Tensor = torch.as_tensor(1.0)
    tau_ac_inv: torch.Tensor = torch.as_tensor(1.0 / 100e-3)
    tau_c_inv: torch.Tensor = torch.as_tensor(1.0 / 100e-3)


class CorrelationSensorState(NamedTuple):
    post_pre: torch.Tensor
    correlation_trace: torch.Tensor
    anti_correlation_trace: torch.Tensor


def correlation_sensor_step(
    z_pre: torch.Tensor,
    z_post: torch.Tensor,
    state: CorrelationSensorState,
    p: CorrelationSensorParameters = CorrelationSensorParameters(),
    dt: float = 0.001,
) -> CorrelationSensorState:
    """Euler integration step of an idealized version of the correlation sensor
    as it is present on the BrainScaleS 2 chips.
    """
    dcorrelation_trace = dt * p.tau_c_inv * (-state.correlation_trace)
    correlation_trace_decayed = (
        state.correlation_trace + (1 - state.post_pre) * dcorrelation_trace
    )

    danti_correlation_trace = dt * p.tau_ac_inv * (-state.anti_correlation_trace)
    anti_correlation_trace_decayed = (
        state.anti_correlation_trace + state.post_pre * danti_correlation_trace
    )

    # compute the pre and post masks based on the current spikes
    pre_spike_mask = pre_mask(state.post_pre, z_pre)
    post_spike_mask = post_mask(state.post_pre, z_post)

    post_pre_new = post_pre_update(state.post_pre, post_spike_mask, pre_spike_mask)
    correlation_trace_new = correlation_trace_decayed + (p.eta_p * pre_spike_mask)
    anti_correlation_trace_new = (
        anti_correlation_trace_decayed + p.eta_m * post_spike_mask
    )

    return CorrelationSensorState(
        post_pre=post_pre_new,
        correlation_trace=correlation_trace_new,
        anti_correlation_trace=anti_correlation_trace_new,
    )


def correlation_based_update(
    ts: int,
    linear_update: torch.nn.Module,
    weights: torch.Tensor,
    correlation_state: CorrelationSensorState,
    learning_rate: float,
    ts_frequency: int,
):
    if ts % ts_frequency == 0:
        (_, input_features, hidden_features) = correlation_state.correlation_trace.shape
        # proposed weight update
        dw = torch.cat(
            (
                correlation_state.correlation_trace.flatten(),
                correlation_state.anti_correlation_trace.flatten(),
            )
        )
        dw = linear_update(dw).detach()
        weights = weights + learning_rate * torch.reshape(
            dw, (hidden_features, input_features)
        )
        # reset correlation traces
        correlation_state.correlation_trace.zero_()
        correlation_state.anti_correlation_trace.zero_()

    return weights
