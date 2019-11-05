import torch
from typing import Tuple

from .lif import LIFState, LIFFeedForwardState, LIFParameters
from .threshhold import threshhold


def lif_mc_step(
    input: torch.Tensor,
    s: LIFState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    g_coupling: torch.Tensor,
    p: LIFParameters = LIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFState]:
    """Computes a single euler-integration step of a LIF multi-compartment
    neuron-model.

    Parameters:
        input (torch.Tensor): the input spikes at the current time step
        s (LIFState): current state of the neuron
        input_weights (torch.Tensor): synaptic weights for incoming spikes
        recurrent_weights (torch.Tensor): synaptic weights for recurrent spikes
        g_coupling (torch.Tensor): conductances between the neuron compartments
        p (LIFParameters): neuron parameters
        dt (float): Integration timestep to use
    """
    # compute voltage
    dv = dt * p.tau_mem_inv * ((p.v_leak - s.v) + s.i) + torch.nn.functional.linear(
        s.v, g_coupling
    )
    v_decayed = s.v + dv
    # compute current updates
    di = -dt * p.tau_syn_inv * s.i
    i_decayed = s.i + di
    # compute new spikes
    z_new = threshhold(v_decayed - p.v_th, p.method, p.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset
    # compute current jumps
    i_new = (
        i_decayed
        + torch.nn.functional.linear(input, input_weights)
        + torch.nn.functional.linear(s.z, recurrent_weights)
    )
    return z_new, LIFState(z_new, v_new, i_new)


def lif_mc_feed_forward_step(
    input: torch.Tensor,
    s: LIFFeedForwardState,
    g_coupling: torch.Tensor,
    p: LIFParameters = LIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFFeedForwardState]:
    """Computes a single euler-integration feed forward step of a LIF multi-compartment
    neuron-model.

    Parameters:
        input (torch.Tensor): the (weighted) input spikes at the current time step
        s (LIFFeedForwardState): current state of the neuron
        g_coupling (torch.Tensor): conductances between the neuron compartments
        p (LIFParameters): neuron parameters
        dt (float): Integration timestep to use
    """
    # compute voltage
    dv = dt * p.tau_mem_inv * ((p.v_leak - s.v) + s.i) + torch.nn.functional.linear(
        s.v, g_coupling
    )
    v_decayed = s.v + dv
    # compute current updates
    di = -dt * p.tau_syn_inv * s.i
    i_decayed = s.i + di
    # compute new spikes
    z_new = threshhold(v_decayed - p.v_th, p.method, p.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset
    # compute current jumps
    i_new = i_decayed + input
    return z_new, LIFFeedForwardState(v_new, i_new)
