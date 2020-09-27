import torch
import torch.jit

from typing import NamedTuple, Tuple


class LIState(NamedTuple):
    """State of a leaky-integrator

    Parameters:
        v (torch.Tensor): membrane voltage
        i (torch.Tensor): input current
    """

    v: torch.Tensor
    i: torch.Tensor


class LIParameters(NamedTuple):
    """Parameters of a leaky integrator

    Parameters:
        tau_syn_inv (torch.Tensor): inverse synaptic time constant
        tau_mem_inv (torch.Tensor): inverse membrane time constant
        v_leak (torch.Tensor): leak potential
        v_reset (torch.Tensor): reset potential
    """

    tau_syn_inv: torch.Tensor = torch.tensor(1.0 / 5e-3)
    tau_mem_inv: torch.Tensor = torch.tensor(1.0 / 1e-2)
    v_leak: torch.Tensor = torch.tensor(0.0)
    v_reset: torch.Tensor = torch.tensor(0.0)


def li_step(
    input: torch.Tensor,
    s: LIState,
    input_weights: torch.Tensor,
    p: LIParameters = LIParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIState]:
    r"""Single euler integration step of a leaky-integrator.
    More specifically it implements a discretized version of the ODE

    .. math::

        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}


    and transition equations

    .. math::
        i = i + w i_{\text{in}}

    Parameters:
        input (torch.Tensor); Input spikes
        s (LIState): state of the leaky integrator
        input_weights (torch.Tensor): weights for incoming spikes
        p (LIParameters): parameters of the leaky integrator
        dt (float): integration timestep to use
    """

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
