import torch

from .threshhold import threshhold
from typing import NamedTuple, Tuple


class LSNNParameters(NamedTuple):
    """"Parameters of an LSNN neuron"""

    tau_syn_inv: torch.Tensor = torch.tensor(1.0 / 5e-3)
    """inverse synaptic time constant"""
    tau_mem_inv: torch.Tensor = torch.tensor(1.0 / 1e-2)
    """inverse membrane time constant"""
    tau_adapt_inv: torch.Tensor = torch.tensor(1.0 / 700)
    """inverse adaptation time constant"""
    v_leak: torch.Tensor = torch.tensor(0.0)
    """leak potential"""
    v_th: torch.Tensor = torch.tensor(1.0)
    """threshhold potential"""
    v_reset: torch.Tensor = torch.tensor(0.0)
    """reset potential"""
    beta: torch.Tensor = torch.tensor(1.8)
    """adaptation constant"""
    method: str = "super"
    alpha: float = 100.0


class LSNNState(NamedTuple):
    z: torch.Tensor
    """recurrent spikes"""
    v: torch.Tensor
    """membrane potential"""
    i: torch.Tensor
    """synaptic input current"""
    b: torch.Tensor
    """threshhold adaptation"""


@torch.jit.script
def lsnn_step(
    input: torch.Tensor,
    s: LSNNState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: LSNNParameters = LSNNParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LSNNState]:
    """Euler integration step for LIF Neuron with threshhold adaptation

    Parameters:
        input (Tensor): the input spikes at the current time step
        s (LSNNState): current state of the lsnn unit
        input_weights (Tensor): synaptic weights for input spikes
        recurrent_weights (Tensor): synaptic weights for recurrent spikes
        p (LSNNParameters): parameters of the lsnn unit
        dt (float): Integration timestep to use
    """
    # compute voltage decay
    dv = dt * p.tau_mem_inv * ((p.v_leak - s.v) + s.i)
    v_decayed = s.v + dv

    # compute current decay
    di = -dt * p.tau_syn_inv * s.i
    i_decayed = s.i + di

    # compute threshhold adaptation update
    db = dt * p.tau_adapt_inv * (p.v_th - s.b)
    b_decayed = s.b + db

    # compute new spikes
    z_new = threshhold(v_decayed - b_decayed, p.method, p.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset
    # compute current jumps
    i_new = (
        i_decayed
        + torch.nn.functional.linear(input, input_weights)
        + torch.nn.functional.linear(s.z, recurrent_weights)
    )

    b_new = b_decayed + z_new * p.tau_adapt_inv * p.beta
    return z_new, LSNNState(z_new, v_new, i_new, b_new)


class LSNNFeedForwardState(NamedTuple):
    """Integration state kept for a lsnn module"""

    v: torch.Tensor
    """membrane potential"""
    i: torch.Tensor
    """synaptic input current"""
    b: torch.Tensor
    """threshhold adaptation"""


@torch.jit.script
def lsnn_feed_forward_step(
    input: torch.Tensor,
    s: LSNNFeedForwardState,
    p: LSNNParameters = LSNNParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LSNNFeedForwardState]:
    """Euler integration step for LIF Neuron with threshhold adaptation

    Parameters:
        input (Tensor): the input spikes at the current time step
        s (LSNNFeedForwardState): current state of the lsnn unit
        p (LSNNParameters): parameters of the lsnn unit
        dt (float): Integration timestep to use
    """
    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - s.v) + s.i)
    v_decayed = s.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * s.i
    i_decayed = s.i + di

    # compute threshhold updates
    db = dt * p.tau_adapt_inv * (p.v_th - s.b)
    b_decayed = s.b + db

    # compute new spikes
    z_new = threshhold(v_decayed - b_decayed, p.method, p.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset
    # compute b update
    b_new = (1 - z_new) * b_decayed + z_new * s.b
    # compute current jumps
    i_new = i_decayed + input

    return z_new, LSNNFeedForwardState(v=v_new, i=i_new, b=b_new)
