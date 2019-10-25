import torch
import torch.jit

from typing import NamedTuple, Tuple


class LIState(NamedTuple):
    v: torch.Tensor
    """membrane voltage"""
    i: torch.Tensor
    """input current"""


class LIParameters(NamedTuple):
    tau_syn_inv: torch.Tensor = torch.tensor(1.0 / 5e-3)
    """inverse synaptic time constant"""
    tau_mem_inv: torch.Tensor = torch.tensor(1.0 / 1e-2)
    """inverse membrane time constant"""
    v_leak: torch.Tensor = torch.tensor(0.0)
    """leak potential"""
    v_reset: torch.Tensor = torch.tensor(0.0)
    """reset potential"""


def li_step(
    input: torch.Tensor,
    s: LIState,
    input_weights: torch.Tensor,
    p: LIParameters = LIParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIState]:
    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - s.v) + s.i)
    v_new = s.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * s.i
    i_decayed = s.i + di

    # compute current jumps
    i_new = i_decayed + torch.nn.functional.linear(input, input_weights)
    return v_new, LIState(v_new, i_new)


# @torch.jit.script
def li_feed_forward_step(
    input: torch.Tensor, s: LIState, p: LIParameters = LIParameters(), dt: float = 0.001
) -> Tuple[torch.Tensor, LIState]:
    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - s.v) + s.i)
    v_new = s.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * s.i
    i_decayed = s.i + di

    # compute current jumps
    i_new = i_decayed + input
    return v_new, LIState(v_new, i_new)
