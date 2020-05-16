import torch
from typing import Tuple

from .lif import (
    LIFState,
    LIFFeedForwardState,
    LIFParameters,
    lif_step,
    lif_feed_forward_step,
)


def lif_mc_step(
    input_tensor: torch.Tensor,
    state: LIFState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    g_coupling: torch.Tensor,
    parameters: LIFParameters = LIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFState]:
    """Computes a single euler-integration step of a LIF multi-compartment
    neuron-model.

    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        s (LIFState): current state of the neuron
        input_weights (torch.Tensor): synaptic weights for incoming spikes
        recurrent_weights (torch.Tensor): synaptic weights for recurrent spikes
        g_coupling (torch.Tensor): conductances between the neuron compartments
        p (LIFParameters): neuron parameters
        dt (float): Integration timestep to use
    """
<<<<<<< HEAD
    state.v = state.v + dt * torch.nn.functional.linear(state.v, g_coupling)
    return lif_step(
        input_tensor, state, input_weights, recurrent_weights, parameters, dt
    )
=======
    # compute voltage
    dv = dt * parameters.tau_mem_inv * (
        (parameters.v_leak - state.v) + state.i
    ) + torch.nn.functional.linear(state.v, g_coupling)
    v_decayed = state.v + dv
    # compute current updates
    di = -dt * parameters.tau_syn_inv * state.i
    i_decayed = state.i + di
    # compute new spikes
    z_new = threshold(v_decayed - parameters.v_th, parameters.method, parameters.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * parameters.v_reset
    # compute current jumps
    i_new = (
        i_decayed
        + torch.nn.functional.linear(input_tensor, input_weights)
        + torch.nn.functional.linear(state.z, recurrent_weights)
    )
    return z_new, LIFState(z_new, v_new, i_new)
>>>>>>> 6b7bb4790156f9744373086e6e8df96dbb0dfeb8


def lif_mc_feed_forward_step(
    input_tensor: torch.Tensor,
    state: LIFFeedForwardState,
    g_coupling: torch.Tensor,
    parameters: LIFParameters = LIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFFeedForwardState]:
    """Computes a single euler-integration feed forward step of a LIF
    multi-compartment neuron-model.

    Parameters:
        input_tensor (torch.Tensor): the (weighted) input spikes at the
                              current time step
        s (LIFFeedForwardState): current state of the neuron
        g_coupling (torch.Tensor): conductances between the neuron compartments
        p (LIFParameters): neuron parameters
        dt (float): Integration timestep to use
    """
<<<<<<< HEAD
    state.v = state.v + dt * torch.nn.functional.linear(state.v, g_coupling)
    return lif_feed_forward_step(input_tensor, state, parameters, dt)
=======
    # compute voltage
    dv = dt * parameters.tau_mem_inv * (
        (parameters.v_leak - state.v) + state.i
    ) + torch.nn.functional.linear(state.v, g_coupling)
    v_decayed = state.v + dv
    # compute current updates
    di = -dt * parameters.tau_syn_inv * state.i
    i_decayed = state.i + di
    # compute new spikes
    z_new = threshold(v_decayed - parameters.v_th, parameters.method, parameters.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * parameters.v_reset
    # compute current jumps
    i_new = i_decayed + input_tensor
    return z_new, LIFFeedForwardState(v_new, i_new)
>>>>>>> 6b7bb4790156f9744373086e6e8df96dbb0dfeb8
