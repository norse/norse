import torch
import torch.jit

from typing import NamedTuple, Tuple


class STDPSensorParameters(NamedTuple):
    eta_p: torch.Tensor = torch.tensor(1.0)
    """correlation state"""
    eta_m: torch.Tensor = torch.tensor(1.0)
    """anti correlation state"""
    tau_ac_inv: torch.Tensor = torch.tensor(1.0 / 100e-3)
    """anti-correlation sensor time constant"""
    tau_c_inv: torch.Tensor = torch.tensor(1.0 / 100e-3)
    """correlation sensor time constant"""


class STDPSensorState(NamedTuple):
    a_pre: torch.Tensor
    a_post: torch.Tensor


@torch.jit.script
def stdp_sensor_step(
    z_pre: torch.Tensor,
    z_post: torch.Tensor,
    s: STDPSensorState,
    p: STDPSensorParameters = STDPSensorParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, STDPSensorState]:
    """Event driven STDP rule."""
    da_pre = p.tau_c_inv * (-s.a_pre)
    a_pre_decayed = s.a_pre + dt * da_pre

    da_post = p.tau_c_inv * (-s.a_post)
    a_post_decayed = s.a_post + dt * da_post

    a_pre_new = a_pre_decayed + z_pre * p.eta_p
    a_post_new = a_post_decayed + z_post * p.eta_m

    dw = z_pre * a_pre_new + z_post * a_post_new
    return dw, STDPSensorState(a_pre_new, a_post_new)
