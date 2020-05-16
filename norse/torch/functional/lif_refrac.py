import torch

from .lif import (
    LIFParameters,
    LIFState,
    LIFFeedForwardState,
    lif_step,
    lif_feed_forward_step,
)
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

    z_new, s_new = lif_step(
        input_tensor, state.lif, input_weights, recurrent_weights, parameters.lif, dt
    )
    v_new = (1 - refrac_mask) * s_new.v + refrac_mask * state.lif.v
    z_new = (1 - refrac_mask) * z_new

    # compute update to refractory counter
    rho_new = (1 - z_new) * torch.nn.functional.relu(
        state.rho - refrac_mask
    ) + z_new * parameters.rho_reset

    return z_new, LIFRefracState(LIFState(z_new, v_new, s_new.i_new), rho_new)


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
    refrac_mask = threshold(state.rho, parameters.lif.method, parameters.lif.alpha)

    z_new, s_new = lif_feed_forward_step(input_tensor, state.lif, parameters.lif, dt)
    v_new = (1 - refrac_mask) * s_new.v + refrac_mask * state.lif.v
    z_new = (1 - refrac_mask) * z_new

    # compute update to refractory counter
    rho_new = (1 - z_new) * torch.nn.functional.relu(
        state.rho - refrac_mask
    ) + z_new * parameters.rho_reset

    return (
        z_new,
        LIFRefracFeedForwardState(LIFFeedForwardState(v_new, s_new.i), rho_new),
    )
