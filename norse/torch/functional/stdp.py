from typing import Callable, NamedTuple


class STDPState(NamedTuple):
    """State of an event driven STDP.

    Parameters:
        a_pre (torch.Tensor): presynaptic STDP state.
        a_post (torch.Tensor): postsynaptic STDP state.
    """

    a_pre: torch.Tensor
    a_post: torch.Tensor


def A_plus_soft(w, w_max, eta_plus):
    return (w_max - w) * eta_plus


def A_plus_hard(w, w_max, eta_plus):
    return heaviside(w_max - w) * eta_plus


def A_minus_soft(w, eta_minus):
    return w * eta_minus


def A_minus_hard(w, eta_minus):
    return heaviside(-w) * eta_minus


def stdp_step(
    z_pre: torch.Tensor,
    z_post: torch.Tensor,
    A_plus: torch.nn.Module,
    A_minus: torch.nn.Module,
    state: STDPState,
    dt: float = 0.001,
) -> Tuple[torch.Tensor, STDPState]:
    """Event driven STDP rule.

    Parameters:
        z_pre (torch.Tensor): pre-synaptic spikes
        z_post (torch.Tensor): post-synaptic spikes
        A_plus (torch.nn.Module): function or pytorch module that parametrises the update
        A_minus (torch.nn.Module): function or pytorch module that parametrises the update
        s (STDPState): state of the STDP sensor
        dt (float): integration time step
    """
    da_pre = p.tau_pre_inv * (-state.a_pre)
    a_pre_decayed = state.a_pre + dt * da_pre

    da_post = p.tau_post_inv * (-state.a_post)
    a_post_decayed = state.a_post + dt * da_post

    a_pre_new = a_pre_decayed + z_pre * p.eta_p
    a_post_new = a_post_decayed + z_post * p.eta_m

    dw = A_plus(w) * z_pre * a_pre_new - A_minus(w) * z_post * a_post_new
    return dw, STDPSensorState(a_pre_new, a_post_new)
