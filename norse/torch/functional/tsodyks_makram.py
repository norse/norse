from typing import NamedTuple, Tuple

import torch


class TsodyksMakramState(NamedTuple):
    """State of the Tsodyks-Makram Model, note
    that we are tracking the input current state separately.

    Parameters:
        u (torch.Tensor): utilization parameter.
        x (torch.Tensor): fraction of remaining available resources.
    """

    u: torch.Tensor
    x: torch.Tensor


class TsodyksMakramParameters(NamedTuple):
    """Parameters of the Tsodyks-Makram Model

    Parameters:
        tau_f_inv (float): facilitation time constant (in ms).
        tau_s_inv (float): synaptic time constant (in ms).
        tau_d_inv (float): depressing time constant (in ms).
        U (float): size of the jump in the utilization variable.
    """

    tau_f_inv: float = 1 / (50.0e-3)
    tau_s_inv: float = 1 / (20.0e-3)
    tau_d_inv: float = 1 / (750.0e-3)
    U: float = 0.45


def stp_step(
    z: torch.Tensor,
    s: TsodyksMakramState,
    p: TsodyksMakramParameters,
    dt: float = 0.001,
) -> Tuple[torch.Tensor, TsodyksMakramState]:
    """Euler integration step for Tsodyks Makram model of STP.

    Reference: http://www.scholarpedia.org/article/Short-term_synaptic_plasticity

    Parameters:
        z (torch.Tensor): Input spikes
        s (TsodyksMakramState): State of the Tsodyks-Makram model
        p (TsodyksMakramParameters): Parameters of the Tsodyks-Makram model
        dt (float): Euler integration timestep
    """
    du = -p.tau_f_inv * s.u
    u_p = s.u + dt * du + p.U * (1 - s.u) * z

    dx = p.tau_d_inv * (1 - s.x)
    x_p = s.x + dt * dx - u_p * s.x * z

    return u_p * s.x * z, TsodyksMakramState(u=u_p, x=x_p)
