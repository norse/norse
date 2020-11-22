from typing import NamedTuple

import torch


class STDPSensorParameters(NamedTuple):
    """Parameters of an STDP sensor as it is used for event driven
    plasticity rules.

    Parameters:
        eta_p (torch.Tensor): correlation state
        eta_m (torch.Tensor): anti correlation state
        tau_ac_inv (torch.Tensor): anti-correlation sensor time constant
        tau_c_inv (torch.Tensor): correlation sensor time constant
    """

    eta_p: torch.Tensor = torch.as_tensor(1.0)
    eta_m: torch.Tensor = torch.as_tensor(1.0)
    tau_ac_inv: torch.Tensor = torch.as_tensor(1.0 / 100e-3)
    tau_c_inv: torch.Tensor = torch.as_tensor(1.0 / 100e-3)


class STDPSensorState(NamedTuple):
    """State of an event driven STDP sensor.

    Parameters:
        a_pre (torch.Tensor): presynaptic STDP sensor state.
        a_post (torch.Tensor): postsynaptic STDP sensor state.
    """

    a_pre: torch.Tensor
    a_post: torch.Tensor


def stdp_sensor_step(
    z_pre: torch.Tensor,
    z_post: torch.Tensor,
    state: STDPSensorState,
    p: STDPSensorParameters = STDPSensorParameters(),
    dt: float = 0.001,
) -> STDPSensorState:
    """Event driven STDP rule.

    Parameters:
        z_pre (torch.Tensor): pre-synaptic spikes
        z_post (torch.Tensor): post-synaptic spikes
        s (STDPSensorState): state of the STDP sensor
        p (STDPSensorParameters): STDP sensor parameters
        dt (float): integration time step
    """
    da_pre = p.tau_c_inv * (-state.a_pre)
    a_pre_decayed = state.a_pre + dt * da_pre

    da_post = p.tau_c_inv * (-state.a_post)
    a_post_decayed = state.a_post + dt * da_post

    a_pre_new = a_pre_decayed + z_pre * p.eta_p
    a_post_new = a_post_decayed + z_post * p.eta_m

    return STDPSensorState(a_pre_new, a_post_new)
