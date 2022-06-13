from typing import Tuple

import torch

from norse.torch.functional.lif import (
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
    p: LIFParameters = LIFParameters(),
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
    z, s = lif_step(
        input_tensor,
        state,
        input_weights,
        recurrent_weights,
        p,
        dt,
    )
    v_new = s.v + dt * torch.nn.functional.linear(s.v, g_coupling)
    return z, LIFState(s.z, v_new, s.i)


def lif_mc_feed_forward_step(
    input_tensor: torch.Tensor,
    state: LIFFeedForwardState,
    g_coupling: torch.Tensor,
    p: LIFParameters = LIFParameters(),
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
    v_new = state.v + dt * torch.nn.functional.linear(state.v, g_coupling)
    return lif_feed_forward_step(
        input_tensor, LIFFeedForwardState(v_new, state.i), p, dt
    )
