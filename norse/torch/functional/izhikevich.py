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


class IzhikevichSpikingBehaviour(NamedTuple):
    """Spiking behaviour of a Izhikevich neuron
    Parameters:
        p (IzhikevichParameters) : parameters of the Izhikevich neuron model
        s (IzhikevichState) : state of the Izhikevich neuron model
    """

    p: IzhikevichParameters
    s: IzhikevichState


p_ts = IzhikevichParameters(a=0.02, b=0.2, c=-65, d=6)
tonic_spiking = IzhikevichSpikingBehaviour(
    p=p_ts, s=IzhikevichState(v=torch.tensor(-70), u=torch.tensor(-70) * p_ts.b)
)

p_ps = IzhikevichParameters(a=0.02, b=0.25, c=-65, d=6)
phasic_spiking = IzhikevichSpikingBehaviour(
    p=p_ps, s=IzhikevichState(v=torch.tensor(-64), u=torch.tensor(-64) * p_ps.b)
)

p_tb = IzhikevichParameters(a=0.02, b=0.2, c=-50, d=2)
tonic_bursting = IzhikevichSpikingBehaviour(
    p=p_tb, s=IzhikevichState(v=torch.tensor(-70), u=torch.tensor(-70) * p_tb.b)
)

p_pb = IzhikevichParameters(a=0.02, b=0.25, c=-55, d=0.05, tau_inv=200)
phasic_bursting = IzhikevichSpikingBehaviour(
    p=p_pb, s=IzhikevichState(v=torch.tensor(-64), u=torch.tensor(-64) * p_pb.b)
)

p_mm = IzhikevichParameters(a=0.02, b=0.2, c=-55, d=4, tau_inv=250)
mixed_mode = IzhikevichSpikingBehaviour(
    p=p_mm, s=IzhikevichState(v=torch.tensor(-70), u=torch.tensor(-70) * p_mm.b)
)

p_sfa = IzhikevichParameters(a=0.01, b=0.2, c=-65, d=8, tau_inv=250)
spike_frequency_adaptation = IzhikevichSpikingBehaviour(
    p=p_sfa, s=IzhikevichState(v=torch.tensor(-70), u=torch.tensor(-70) * p_sfa.b)
)

p_c1e = IzhikevichParameters(a=0.02, b=-0.1, c=-55, d=6, mn=4.1, bias=108, tau_inv=250)
class_1_exc = IzhikevichSpikingBehaviour(
    p=p_c1e, s=IzhikevichState(v=torch.tensor(-60), u=torch.tensor(-60) * p_c1e.b)
)

p_c2e = IzhikevichParameters(a=0.2, b=0.26, c=-65, d=0, tau_inv=250)
class_2_exc = IzhikevichSpikingBehaviour(
    p=p_c2e, s=IzhikevichState(v=torch.tensor(-64), u=torch.tensor(-64) * p_c2e.b)
)

p_sl = IzhikevichParameters(a=0.02, b=0.2, c=-65, d=6, tau_inv=250)
spike_latency = IzhikevichSpikingBehaviour(
    p=p_sl, s=IzhikevichState(v=torch.tensor(-70), u=torch.tensor(-70) * p_sl.b)
)

p_so = IzhikevichParameters(a=0.05, b=0.26, c=-60, d=0, tau_inv=250)
subthreshold_oscillation = IzhikevichSpikingBehaviour(
    p=p_so, s=IzhikevichState(v=torch.tensor(-62), u=torch.tensor(-62) * p_so.b)
)

p_r = IzhikevichParameters(a=0.1, b=0.26, c=-60, d=-1, tau_inv=250)
resonator = IzhikevichSpikingBehaviour(
    p=p_r, s=IzhikevichState(v=torch.tensor(-62), u=torch.tensor(-62) * p_r.b)
)

p_i = IzhikevichParameters(a=0.02, b=-0.1, c=-55, d=6, mn=4.1, bias=108, tau_inv=250)
integrator = IzhikevichSpikingBehaviour(
    p=p_i, s=IzhikevichState(v=torch.tensor(-60), u=torch.tensor(-60) * p_i.b)
)

p_rs = IzhikevichParameters(a=0.03, b=0.25, c=-60, d=4, tau_inv=200)
rebound_spike = IzhikevichSpikingBehaviour(
    p=p_rs, s=IzhikevichState(v=torch.tensor(-64), u=torch.tensor(-64) * p_rs.b)
)

p_rb = IzhikevichParameters(a=0.03, b=0.25, c=-52, d=0, tau_inv=200)
rebound_burst = IzhikevichSpikingBehaviour(
    p=p_rb, s=IzhikevichState(v=torch.tensor(-64), u=torch.tensor(-64) * p_rb.b)
)

p_tv = IzhikevichParameters(a=0.03, b=0.25, c=-60, d=4, tau_inv=250)
threshhold_variability = IzhikevichSpikingBehaviour(
    p=p_tv, s=IzhikevichState(v=torch.tensor(-64), u=torch.tensor(-64) * p_tv.b)
)

p_b = IzhikevichParameters(a=0.1, b=0.26, c=-60, d=0, tau_inv=250)
bistability = IzhikevichSpikingBehaviour(
    p=p_b, s=IzhikevichState(v=torch.tensor(-61), u=torch.tensor(-61) * p_b.b)
)

p_dap = IzhikevichParameters(a=1.0, b=0.2, c=-60, d=-21, tau_inv=100)
dap = IzhikevichSpikingBehaviour(
    p=p_dap, s=IzhikevichState(v=torch.tensor(-70), u=torch.tensor(-70) * p_dap.b)
)

p_a = IzhikevichParameters(a=0.02, b=1.0, c=-55, d=4, tau_inv=500)
accomodation = IzhikevichSpikingBehaviour(
    p=p_a, s=IzhikevichState(v=torch.tensor(-65), u=torch.tensor(-16))
)

p_iis = IzhikevichParameters(a=-0.02, b=-1.0, c=-60, d=8, tau_inv=250)
inhibition_induced_spiking = IzhikevichSpikingBehaviour(
    p=p_iis, s=IzhikevichState(v=torch.tensor(-63.8), u=torch.tensor(-63.8) * p_iis.b)
)

p_iib = IzhikevichParameters(a=-0.026, b=-1.0, c=-45, d=-2, tau_inv=250)
inhibition_induced_bursting = IzhikevichSpikingBehaviour(
    p=p_iib, s=IzhikevichState(v=torch.tensor(-63.8), u=torch.tensor(-63.8) * p_iib.b)
)


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
