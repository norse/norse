import torch
from typing import NamedTuple, Tuple
from norse.torch.functional.threshold import threshold


class IAFParameters(NamedTuple):
    """Parametrization of an integrate-and-fire neuron

    Parameters:
        v_th (torch.Tensor): threshold potential in mV
        v_reset (torch.Tensor): reset potential in mV
        method (str): method to determine the spike threshold
                      (relevant for surrogate gradients)
        alpha (float): hyper parameter to use in surrogate gradient computation
    """

    v_th: torch.Tensor = torch.as_tensor(1.0)
    v_reset: torch.Tensor = torch.as_tensor(0.0)
    method: str = "super"
    alpha: float = torch.as_tensor(100.0)


class IAFState(NamedTuple):
    """State of an integrate-and-fire neuron

    Parameters:
        z (torch.Tensor): recurrent spikes
        v (torch.Tensor): membrane potential
    """

    z: torch.Tensor
    v: torch.Tensor


class IAFFeedForwardState(NamedTuple):
    """State of a feed forward integrate-and-fire neuron

    Parameters:
        v (torch.Tensor): membrane potential
    """

    v: torch.Tensor


def iaf_step(
    input_spikes: torch.Tensor,
    state: IAFState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: IAFParameters = IAFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, IAFState]:
    v_new = (
        state.v
        + torch.nn.functional.linear(input_spikes, input_weights)
        + torch.nn.functional.linear(state.z, recurrent_weights)
    )
    # compute new spikes
    z_new = threshold(v_new - p.v_th, p.method, p.alpha)
    # compute reset
    v_new = (1 - z_new) * v_new + z_new * p.v_reset
    return z_new, IAFState(z_new, v_new)


def iaf_feed_forward_step(
    input_spikes: torch.Tensor,
    state: IAFFeedForwardState,
    p: IAFParameters = IAFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, IAFFeedForwardState]:
    r"""Feedforward step of an integrate-and-fire neuron, computing a single step

    .. math::
        \dot{v} = v

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}})

    and transition equation

    .. math::
        v = (1-z) v + z v_{\text{reset}}

    Parameters:
        input_spikes (torch.Tensor): the input spikes at the current time step
        state (IAFFeedForwardState): current state of the LIF neuron
        p (IAFParameters): parameters of a leaky integrate and fire neuron
        dt (float): Integration timestep to use (unused, but added for compatibility)
    """
    # compute current jumps
    v_new = state.v + input_spikes
    # compute new spikes
    z_new = threshold(v_new - p.v_th, p.method, p.alpha)
    # compute reset
    v_new = (1 - z_new) * v_new + z_new * p.v_reset

    return z_new, IAFFeedForwardState(v=v_new)
