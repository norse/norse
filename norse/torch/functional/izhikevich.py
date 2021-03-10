import torch
from typing import NamedTuple, Tuple

from norse.torch.functional.threshold import threshold


class IzhikevichParameters(NamedTuple):
    """Parametrization of av Izhikevich neuron
    Parameters:
        a (float): time scale of the recovery variable u. Smaller values result in slower recovery in 1/ms
        b (float): sensitivity of the recovery variable u to the subthreshold fluctuations of the membrane potential v. Greater values couple v and u more strongly resulting in possible subthreshold oscillations and low-threshold spiking dynamics
        c (float): after-spike reset value of the membrane potential in mV
        d (float): after-spike reset of the recovery variable u caused by slow high-threshold Na+ and K+ conductances in mV
        sq (float): constant of the v squared variable in mV/ms
        mn (float): constant of the v variable in 1/ms
        bias (float): bias constant in mV/ms
        v_th (torch.Tensor): threshold potential in mV
        tau_inv (float) : inverse time constant in 1/ms
        method (str): method to determine the spike threshold
                      (relevant for surrogate gradients)
        alpha (float): hyper parameter to use in surrogate gradient computation
    """
    
    a: float
    b: float
    c: float
    d: float
    sq: float = 0.04
    mn: float = 5
    bias: float = 140
    v_th: float = 30
    tau_inv: float = 250
    method: str = "super"
    alpha: float = 100.0


class IzhikevichState(NamedTuple):
    """State of a Izhikevich neuron
    Parameters:
        v (torch.Tensor): membrane potential
        u (torch.Tensor): membrane recovery variable
    """
    
    v: torch.Tensor
    u: torch.Tensor


def tonic_spiking(shape) -> Tuple[IzhikevichState, IzhikevichParameters]:
    
    p = IzhikevichParameters(a=0.02, b=0.2, c=-65, d=6)
    s = IzhikevichState(v=-70 * torch.ones(shape), u=-70 * p.b * torch.ones(shape))
    return s, p


def phasic_spiking(shape) -> Tuple[IzhikevichState, IzhikevichParameters]:
    p = IzhikevichParameters(a=0.02, b=0.25, c=-65, d=6)
    s = IzhikevichState(v=-64 * torch.ones(shape), u=-64 * p.b * torch.ones(shape))
    return s, p


def tonic_bursting(shape) -> Tuple[IzhikevichState, IzhikevichParameters]:
    p = IzhikevichParameters(a=0.02, b=0.2, c=-50, d=2)
    s = IzhikevichState(v=-70 * torch.ones(shape), u=-70 * p.b * torch.ones(shape))
    return s, p


def phasic_bursting(shape) -> Tuple[IzhikevichState, IzhikevichParameters]:
    p = IzhikevichParameters(a=0.02, b=0.25, c=-55, d=0.05, tau_inv=200)
    s = IzhikevichState(v=-64 * torch.ones(shape), u=-64 * p.b * torch.ones(shape))
    return s, p


def mixed_mode(shape) -> Tuple[IzhikevichState, IzhikevichParameters]:
    p = IzhikevichParameters(a=0.02, b=0.2, c=-55, d=4, tau_inv=250)
    s = IzhikevichState(v=-70 * torch.ones(shape), u=-70 * p.b * torch.ones(shape))
    return s, p


def spike_frequency_adaptation(shape) -> Tuple[IzhikevichState, IzhikevichParameters]:
    p = IzhikevichParameters(a=0.01, b=0.2, c=-65, d=8, tau_inv=250)
    s = IzhikevichState(v=-70 * torch.ones(shape), u=-70 * p.b * torch.ones(shape))
    return s, p


def class_1_exc(shape) -> Tuple[IzhikevichState, IzhikevichParameters]:
    p = IzhikevichParameters(a=0.02, b=-0.1, c=-55, d=6, mn=4.1, bias=108, tau_inv=250)
    s = IzhikevichState(v=-60 * torch.ones(shape), u=-60 * p.b * torch.ones(shape))
    return s, p


def class_2_exc(shape) -> Tuple[IzhikevichState, IzhikevichParameters]:
    p = IzhikevichParameters(a=0.2, b=0.26, c=-65, d=0, tau_inv=250)
    s = IzhikevichState(v=-64 * torch.ones(shape), u=-64 * p.b * torch.ones(shape))
    return s, p


def spike_latency(shape) -> Tuple[IzhikevichState, IzhikevichParameters]:
    p = IzhikevichParameters(a=0.02, b=0.2, c=-65, d=6, tau_inv=250)
    s = IzhikevichState(v=-70 * torch.ones(shape), u=-70 * p.b * torch.ones(shape))
    return s, p


def subthreshold_oscillation(shape) -> Tuple[IzhikevichState, IzhikevichParameters]:
    p = IzhikevichParameters(a=0.05, b=0.26, c=-60, d=0, tau_inv=250)
    s = IzhikevichState(v=-62 * torch.ones(shape), u=-62 * p.b * torch.ones(shape))
    return s, p


def resonator(shape) -> Tuple[IzhikevichState, IzhikevichParameters]:
    p = IzhikevichParameters(a=0.1, b=0.26, c=-60, d=-1, tau_inv=250)
    s = IzhikevichState(v=-62 * torch.ones(shape), u=-62 * p.b * torch.ones(shape))
    return s, p


def integrator(shape) -> Tuple[IzhikevichState, IzhikevichParameters]:
    p = IzhikevichParameters(a=0.02, b=-0.1, c=-55, d=6, mn=4.1, bias=108, tau_inv=250)
    s = IzhikevichState(v=-60 * torch.ones(shape), u=-60 * p.b * torch.ones(shape))
    return s, p


def rebound_spike(shape) -> Tuple[IzhikevichState, IzhikevichParameters]:
    p = IzhikevichParameters(a=0.03, b=0.25, c=-60, d=4, tau_inv=200)
    s = IzhikevichState(v=-64 * torch.ones(shape), u=-64 * p.b * torch.ones(shape))
    return s, p


def rebound_burst(shape) -> Tuple[IzhikevichState, IzhikevichParameters]:
    p = IzhikevichParameters(a=0.03, b=0.25, c=-52, d=0, tau_inv=200)
    s = IzhikevichState(v=-64 * torch.ones(shape), u=-64 * p.b * torch.ones(shape))
    return s, p


def threshhold_variability(shape) -> Tuple[IzhikevichState, IzhikevichParameters]:
    p = IzhikevichParameters(a=0.03, b=0.25, c=-60, d=4, tau_inv=250)
    s = IzhikevichState(v=-64 * torch.ones(shape), u=-64 * p.b * torch.ones(shape))
    return s, p


def bistability(shape) -> Tuple[IzhikevichState, IzhikevichParameters]:
    p = IzhikevichParameters(a=0.1, b=0.26, c=-60, d=0, tau_inv=250)
    s = IzhikevichState(v=-61 * torch.ones(shape), u=-61 * p.b * torch.ones(shape))
    return s, p


def dap(shape) -> Tuple[IzhikevichState, IzhikevichParameters]:
    p = IzhikevichParameters(a=1.0, b=0.2, c=-60, d=-21, tau_inv=100)
    s = IzhikevichState(v=-70 * torch.ones(shape), u=-70 * p.b * torch.ones(shape))
    return s, p


def accomodation(shape) -> Tuple[IzhikevichState, IzhikevichParameters]:
    p = IzhikevichParameters(a=0.02, b=1.0, c=-55, d=4, tau_inv=500)
    s = IzhikevichState(v=-65 * torch.ones(shape), u=-16 * torch.ones(shape))
    return s, p


def inhibition_induced_spiking(shape) -> Tuple[IzhikevichState, IzhikevichParameters]:
    p = IzhikevichParameters(a=-0.02, b=-1.0, c=-60, d=8, tau_inv=250)
    s = IzhikevichState(v=-63.8 * torch.ones(shape), u=-63.8 * p.b * torch.ones(shape))
    return s, p


def inhibition_induced_bursting(shape) -> Tuple[IzhikevichState, IzhikevichParameters]:
    p = IzhikevichParameters(a=-0.026, b=-1.0, c=-45, d=-2, tau_inv=250)
    s = IzhikevichState(v=-63.8 * torch.ones(shape), u=-63.8 * p.b * torch.ones(shape))
    return s, p


def izhikevich_step(
    input_current: torch.Tensor,
    s: IzhikevichState,
    p: IzhikevichParameters,
    dt: float = 0.001,
):
    v_ = s.v + p.tau_inv * dt * (
        p.sq * s.v ** 2 + p.mn * s.v + p.bias - s.u + input_current
    )
    u_ = s.u + p.tau_inv * dt * p.a * (p.b * s.v - s.u)
    z_ = threshold(v_ - p.v_th, p.method, p.alpha)
    v_ = (1 - z_) * v_ + z_ * p.c
    u_ = (1 - z_) * u_ + z_ * (u_ + p.d)
    return z_, IzhikevichState(v_, u_)
