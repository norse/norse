import torch

from .lif_refrac import LIFRefracState, LIFRefracFeedForwardState, LIFRefracParameters
from .lif import LIFParameters, LIFState, LIFFeedForwardState
from .threshhold import threshhold

from typing import Tuple


def lif_mc_refrac_step(
    input: torch.Tensor,
    s: LIFRefracState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    g_coupling: torch.Tensor,
    p: LIFRefracParameters = LIFRefracParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFRefracState]:
    # compute whether neurons are refractory or not
    refrac_mask = threshhold(s.rho, p.lif.method, p.lif.alpha)
    # compute voltage
    dv = (1 - refrac_mask) * dt * p.lif.tau_mem_inv * (
        (p.lif.v_leak - s.lif.v) + s.lif.i
    ) + torch.nn.functional.linear(s.lif.v, g_coupling)
    v_decayed = s.lif.v + dv

    # compute current updates
    di = -dt * p.lif.tau_syn_inv * s.lif.i
    i_decayed = s.lif.i + di

    # compute new spikes
    z_new = threshhold(v_decayed - p.lif.v_th, p.lif.method, p.lif.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.lif.v_reset

    # compute current jumps
    i_new = (
        i_decayed
        + torch.nn.functional.linear(input, input_weights)
        + torch.nn.functional.linear(s.lif.z, recurrent_weights)
    )

    # compute update to refractory counter
    rho_new = (1 - z_new) * torch.nn.functional.relu(
        s.rho - refrac_mask
    ) + z_new * p.rho_reset

    return z_new, LIFRefracState(LIFState(z_new, v_new, i_new), rho_new)


def lif_mc_refrac_feed_forward_step(
    input: torch.Tensor,
    s: LIFRefracState,
    g_coupling: torch.Tensor,
    p: LIFRefracParameters = LIFRefracParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFRefracState]:
    # compute whether neurons are refractory or not
    refrac_mask = threshhold(s.rho, p.lif.method, p.lif.alpha)
    # compute voltage
    dv = (1 - refrac_mask) * dt * p.lif.tau_mem_inv * (
        (p.lif.v_leak - s.lif.v) + s.lif.i
    ) + torch.nn.functional.linear(s.lif.v, g_coupling)
    v_decayed = s.lif.v + dv

    # compute current updates
    di = -dt * p.lif.tau_syn_inv * s.lif.i
    i_decayed = s.lif.i + di

    # compute new spikes
    z_new = threshhold(v_decayed - p.lif.v_th, p.lif.method, p.lif.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.lif.v_reset

    # compute current jumps
    i_new = i_decayed + input

    # compute update to refractory counter
    rho_new = (1 - z_new) * torch.nn.functional.relu(
        s.rho - refrac_mask
    ) + z_new * p.rho_reset

    return z_new, LIFRefracFeedForwardState(LIFFeedForwardState(v_new, i_new), rho_new)
