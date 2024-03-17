import torch
from typing import NamedTuple, Tuple, overload

from norse.torch.functional.lif_adex import (
    LIFAdExParameters,
    LIFAdExState,
    LIFAdExFeedForwardState,
    lif_adex_step,
    lif_adex_feed_forward_step,
)

from norse.torch.functional.threshold import threshold


class LIFAdExRefracParameters(NamedTuple):
    lif_adex: LIFAdExParameters = LIFAdExParameters()
    rho_reset: torch.Tensor = torch.as_tensor(5.0)


class LIFAdExRefracState(NamedTuple):
    """State of a LIFAdExRefrac neuron with absolute refractory period.

    Parameters:
        lif_adex (LIFState): state of the LIFAdEx neuron integration
        rho (torch.Tensor): refractory state (count towards zero)
    """

    lif_adex: LIFAdExState
    rho: torch.Tensor


class LIFAdExRefracFeedForwardState(NamedTuple):
    """State of a feed forward LIFAdExRefrac neuron

    Parameters:
        lif_adex: state of the feed forward LIFAdEx
                                   neuron integration
    """

    lif_adex: LIFAdExFeedForwardState
    rho: torch.Tensor


@overload
def compute_refractore_update(
    state: LIFAdExRefracState,
    z_new: torch.Tensor,
    v_new: torch.Tensor,
    p: LIFAdExRefracParameters = LIFAdExRefracParameters(),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


@overload
def compute_refractore_update(
    state: LIFAdExRefracFeedForwardState,
    z_new: torch.Tensor,
    v_new: torch.Tensor,
    p: LIFAdExRefracParameters = LIFAdExRefracParameters(),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


def compute_refractory_update(
    state,
    z_new: torch.Tensor,
    v_new: torch.Tensor,
    p: LIFAdExRefracParameters = LIFAdExRefracParameters(),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the refractory update.
    Parameters:
        state (LIFAdExRefracState): Initial state of the refractory neuron.
        z_new (torch.Tensor): New spikes that were generated.
        v_new (torch.Tensor): New voltage after the lif update step.
        p (torch.Tensor): Refractoryp.
    """
    refrac_mask = threshold(state.rho, p.lif_adex.method, p.lif_adex.alpha)
    v_new = (1 - refrac_mask) * v_new + refrac_mask * state.lif_adex.v
    z_new = (1 - refrac_mask) * z_new

    rho_new = (1 - z_new) * torch.nn.functional.relu(
        state.rho - refrac_mask
    ) + z_new * p.rho_reset
    return v_new, z_new, rho_new


def lif_adex_refrac_step(
    input_tensor: torch.Tensor,
    state: LIFAdExRefracState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: LIFAdExRefracParameters = LIFAdExRefracParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFAdExRefracState]:
    r"""Computes a single euler-integration step of a recurrently connected
     LIFAdEx neuron-model with a refractory period.

    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        s (LIFAdExRefracState): state at the current time step
        input_weights (torch.Tensor): synaptic weights for incoming spikes
        recurrent_weights (torch.Tensor): synaptic weights for recurrent spikes
        p (LIFAdExRefracParameters): parameters of the LIFAdEx neuron
        dt (float): Integration timestep to use
    """
    z_new, s_new = lif_adex_step(
        input_tensor, state.lif_adex, input_weights, recurrent_weights, p.lif_adex, dt
    )
    v_new, z_new, rho_new = compute_refractory_update(state, z_new, s_new.v, p)

    return z_new, LIFAdExRefracState(
        LIFAdExState(z_new, v_new, s_new.i, s_new.a), rho_new
    )


def lif_adex_refrac_feed_forward_step(
    input_tensor: torch.Tensor,
    state: LIFAdExRefracFeedForwardState,
    p: LIFAdExRefracParameters = LIFAdExRefracParameters(),
    dt: float = 0.001,
) -> Tuple[torch.tensor, LIFAdExRefracFeedForwardState]:
    r"""Computes a single euler-integration step of a feed forward
     LIFAdEx neuron-model with a refractory period.

    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        s (LIFAdExRefracFeedForwardState): state at the current time step
        p (LIFAdExRefracParameters): parameters of the LIFAdEx neuron
        dt (float): Integration timestep to use
    """
    z_new, s_new = lif_adex_feed_forward_step(
        input_tensor, state.lif_adex, p.lif_adex, dt
    )
    v_new, z_new, rho_new = compute_refractory_update(state, z_new, s_new.v, p)

    return (
        z_new,
        LIFAdExRefracFeedForwardState(
            LIFAdExFeedForwardState(v_new, s_new.i, s_new.a), rho_new
        ),
    )
