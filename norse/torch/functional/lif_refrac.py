import torch

from .lif import LIFParameters, LIFState, LIFFeedForwardState
from .threshold import threshold

from typing import NamedTuple, Tuple


class LIFRefracState(NamedTuple):
    """State of a LIF neuron with absolute refractory period.

    Parameters:
        lif (LIFState): state of the LIF neuron integration
        rho (torch.Tensor): refractory state (count towards zero)
    """

    lif: LIFState
    rho: torch.Tensor


class LIFRefracParameters(NamedTuple):
    """Parameters of a LIF neuron with absolute refractory period.

    Parameters:
        lif (LIFParameters): parameters of the LIF neuron integration
        rho (torch.Tensor): refractory state (count towards zero)
    """

    lif: LIFParameters = LIFParameters()
    rho_reset: torch.Tensor = torch.as_tensor(5.0)


def lif_refrac_step(
    input_tensor: torch.Tensor,
    state: LIFRefracState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    parameters: LIFRefracParameters = LIFRefracParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFRefracState]:
    r"""Computes a single euler-integration step of a recurrently connected
     LIF neuron-model with a refractory period.

    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        s (LIFRefracState): state at the current time step
        input_weights (torch.Tensor): synaptic weights for incoming spikes
        recurrent_weights (torch.Tensor): synaptic weights for recurrent spikes
        p (LIFRefracParameters): parameters of the lif neuron
        dt (float): Integration timestep to use
    """
    refrac_mask = threshold(state.rho, parameters.lif.method, parameters.lif.alpha)

    # compute voltage updates
    dv = (
        (1 - refrac_mask)
        * dt
        * parameters.lif.tau_mem_inv
        * ((parameters.lif.v_leak - state.lif.v) + state.lif.i)
    )
    v_decayed = state.lif.v + dv

    # compute current updates
    di = -dt * parameters.lif.tau_syn_inv * state.lif.i
    i_decayed = state.lif.i + di

    # compute new spikes
    z_new = threshold(v_decayed - parameters.lif.v_th, parameters.lif.method, parameters.lif.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * parameters.lif.v_reset
    # compute current jumps
    i_new = (
        i_decayed
        + torch.nn.functional.linear(input_tensor, input_weights)
        + torch.nn.functional.linear(state.lif.z, recurrent_weights)
    )

    # compute update to refractory counter
    rho_new = (1 - z_new) * torch.nn.functional.relu(
        state.rho - refrac_mask
    ) + z_new * parameters.rho_reset

    return z_new, LIFRefracState(LIFState(z_new, v_new, i_new), rho_new)


class LIFRefracFeedForwardState(NamedTuple):
    """State of a feed forward LIF neuron with absolute refractory period.

    Parameters:
        lif (LIFFeedForwardState): state of the feed forward LIF
                                   neuron integration
        rho (torch.Tensor): refractory state (count towards zero)
    """

    lif: LIFFeedForwardState
    rho: torch.Tensor


def lif_refrac_feed_forward_step(
    input_tensor: torch.Tensor,
    state: LIFRefracFeedForwardState,
    parameters: LIFRefracParameters = LIFRefracParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFRefracFeedForwardState]:
    r"""Computes a single euler-integration step of a feed forward
     LIF neuron-model with a refractory period.

    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        s (LIFRefracFeedForwardState): state at the current time step
        p (LIFRefracParameters): parameters of the lif neuron
        dt (float): Integration timestep to use
    """
    rho_mask = threshold(state.rho, parameters.lif.method, parameters.lif.alpha)

    # compute voltage updates
    dv = (1 - rho_mask) * dt * parameters.lif.tau_mem_inv * ((parameters.lif.v_leak - state.lif.v) + state.lif.i)
    v_decayed = state.lif.v + dv

    # compute current updates
    di = -dt * parameters.lif.tau_syn_inv * state.lif.i
    i_decayed = state.lif.i + di

    # compute new spikes
    z_new = threshold(v_decayed - parameters.lif.v_th, parameters.lif.method, parameters.lif.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * parameters.lif.v_reset
    # compute current jumps
    i_new = i_decayed + input_tensor

    # compute update to refractory counter
    rho_new = (1 - z_new) * torch.nn.functional.relu(
        state.rho - rho_mask
    ) + z_new * parameters.rho_reset
    return z_new, LIFRefracFeedForwardState(LIFFeedForwardState(v_new, i_new), rho_new)
