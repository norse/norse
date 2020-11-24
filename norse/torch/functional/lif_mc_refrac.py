from typing import Tuple

import torch

from norse.torch.functional.lif_refrac import LIFRefracState, LIFRefracFeedForwardState
from norse.torch.functional.lif_refrac import LIFRefracParameters
from norse.torch.functional.lif import LIFState, LIFFeedForwardState
from norse.torch.functional.threshold import threshold


def lif_mc_refrac_step(
    input_tensor: torch.Tensor,
    state: LIFRefracState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    g_coupling: torch.Tensor,
    p: LIFRefracParameters = LIFRefracParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFRefracState]:
    # compute whether neurons are refractory or not
    refrac_mask = threshold(state.rho, p.lif.method, p.lif.alpha)
    # compute voltage
    dv = (1 - refrac_mask) * dt * p.lif.tau_mem_inv * (
        (p.lif.v_leak - state.lif.v) + state.lif.i
    ) + torch.nn.functional.linear(state.lif.v, g_coupling)
    v_decayed = state.lif.v + dv

    # compute current updates
    di = -dt * p.lif.tau_syn_inv * state.lif.i
    i_decayed = state.lif.i + di

    # compute new spikes
    z_new = threshold(v_decayed - p.lif.v_th, p.lif.method, p.lif.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.lif.v_reset

    # compute current jumps
    i_new = (
        i_decayed
        + torch.nn.functional.linear(input_tensor, input_weights)
        + torch.nn.functional.linear(state.lif.z, recurrent_weights)
    )

    # compute update to refractory counter
    rho_new = (1 - z_new) * torch.nn.functional.relu(
        state.rho - refrac_mask
    ) + z_new * p.rho_reset

    return z_new, LIFRefracState(LIFState(z_new, v_new, i_new), rho_new)


def lif_mc_refrac_feed_forward_step(
    input_tensor: torch.Tensor,
    state: LIFRefracFeedForwardState,
    g_coupling: torch.Tensor,
    p: LIFRefracParameters = LIFRefracParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFRefracFeedForwardState]:
    # compute whether neurons are refractory or not
    refrac_mask = threshold(state.rho, p.lif.method, p.lif.alpha)
    # compute voltage
    dv = (1 - refrac_mask) * dt * p.lif.tau_mem_inv * (
        (p.lif.v_leak - state.lif.v) + state.lif.i
    ) + torch.nn.functional.linear(state.lif.v, g_coupling)
    v_decayed = state.lif.v + dv

    # compute current updates
    di = -dt * p.lif.tau_syn_inv * state.lif.i
    i_decayed = state.lif.i + di

    # compute new spikes
    z_new = threshold(v_decayed - p.lif.v_th, p.lif.method, p.lif.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.lif.v_reset

    # compute current jumps
    i_new = i_decayed + input_tensor

    # compute update to refractory counter
    rho_new = (1 - z_new) * torch.nn.functional.relu(
        state.rho - refrac_mask
    ) + z_new * p.rho_reset

    return z_new, LIFRefracFeedForwardState(LIFFeedForwardState(v_new, i_new), rho_new)
