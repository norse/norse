from typing import NamedTuple, Tuple, overload

import torch

from norse.torch.functional.lif import (
    LIFParameters,
    LIFState,
    LIFFeedForwardState,
    lif_step,
    lif_step_sparse,
    lif_feed_forward_step,
)
from norse.torch.functional.threshold import threshold


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


class LIFRefracFeedForwardState(NamedTuple):
    """State of a feed forward LIF neuron with absolute refractory period.

    Parameters:
        lif (LIFFeedForwardState): state of the feed forward LIF
                                   neuron integration
        rho (torch.Tensor): refractory state (count towards zero)
    """

    lif: LIFFeedForwardState
    rho: torch.Tensor


@overload
def compute_refractory_update(
    state: LIFRefracState,
    z_new: torch.Tensor,
    v_new: torch.Tensor,
    p: LIFRefracParameters = LIFRefracParameters(),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...  # pragma: no cover


@overload
def compute_refractory_update(
    state: LIFRefracFeedForwardState,
    z_new: torch.Tensor,
    v_new: torch.Tensor,
    p: LIFRefracParameters = LIFRefracParameters(),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...  # pragma: no cover


def compute_refractory_update(
    state,
    z_new: torch.Tensor,
    v_new: torch.Tensor,
    p: LIFRefracParameters = LIFRefracParameters(),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the refractory update.

    Parameters:
        state (LIFRefracState): Initial state of the refractory neuron.
        z_new (torch.Tensor): New spikes that were generated.
        v_new (torch.Tensor): New voltage after the lif update step.
        p (LIFRefracParameters): Refractory Parameters.
    """
    refrac_mask = threshold(state.rho, p.lif.method, p.lif.alpha)
    v_new = (1 - refrac_mask) * v_new + refrac_mask * state.lif.v
    z_new = (1 - refrac_mask) * z_new

    # compute update to refractory counter
    rho_new = (1 - z_new) * torch.nn.functional.relu(
        state.rho - refrac_mask
    ) + z_new * p.rho_reset

    return v_new, z_new, rho_new


def lif_refrac_step_sparse(
    input_tensor: torch.Tensor,
    state: LIFRefracState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: LIFRefracParameters = LIFRefracParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFRefracState]:
    z_sparse, s_new = lif_step_sparse(
        input_tensor, state.lif, input_weights, recurrent_weights, p.lif, dt
    )
    v_new, z_sparse, rho_new = compute_refractory_update(state, z_sparse, s_new.v, p)
    return z_sparse, LIFRefracState(LIFState(z_sparse, v_new, s_new.i), rho_new)


def lif_refrac_step(
    input_tensor: torch.Tensor,
    state: LIFRefracState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: LIFRefracParameters = LIFRefracParameters(),
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
    z_new, s_new = lif_step(
        input_tensor, state.lif, input_weights, recurrent_weights, p.lif, dt
    )
    v_new, z_new, rho_new = compute_refractory_update(state, z_new, s_new.v, p)

    return z_new, LIFRefracState(LIFState(z_new, v_new, s_new.i), rho_new)


def lif_refrac_feed_forward_step(
    input_tensor: torch.Tensor,
    state: LIFRefracFeedForwardState,
    p: LIFRefracParameters = LIFRefracParameters(),
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
    z_new, s_new = lif_feed_forward_step(input_tensor, state.lif, p.lif, dt)
    v_new, z_new, rho_new = compute_refractory_update(state, z_new, s_new.v, p)

    return (
        z_new,
        LIFRefracFeedForwardState(LIFFeedForwardState(v_new, s_new.i), rho_new),
    )
