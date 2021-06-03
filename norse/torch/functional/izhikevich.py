import torch
import matplotlib.pyplot as plt
import numpy as np
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


class IzhikevichRecurrentState(NamedTuple):
    """State of a Izhikevich neuron
    Parameters:
        v (torch.Tensor): membrane potential
        u (torch.Tensor): membrane recovery variable
    """

    z: torch.Tensor
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


def createIzhikevichSpikingBehaviour(
    a: float,
    b: float,
    c: float,
    d: float,
    v_rest: float,
    u_rest: float,
    tau_inv: float = 250,
    current: float = 1,
    print: bool = False,
    time_print: int = 250,
    timestep_print: float = 0.1,
) -> IzhikevichSpikingBehaviour:
    """
    A function that allows for the creation of custom Izhikevich neurons models, as well as a visualization of their behaviour on a 250 ms time window
    Parameters:
        a (float): time scale of the recovery variable u. Smaller values result in slower recovery in 1/ms
        b (float): sensitivity of the recovery variable u to the subthreshold fluctuations of the membrane potential v. Greater values couple v and u more strongly resulting in possible subthreshold oscillations and low-threshold spiking dynamics
        c (float): after-spike reset value of the membrane potential in mV
        d (float): after-spike reset of the recovery variable u caused by slow high-threshold Na+ and K+ conductances in mV
        v_rest (float): resting value of the v variable in mV
        u_rest (float): resting value of the u variable
        tau_inv (float) : inverse time constant in 1/ms
        current (float) : input current
        time_print (float) : size of the time window in ms
        timestep_print (float) : timestep of the simulation in ms
    """
    params = IzhikevichParameters(a=a, b=b, c=c, d=d, tau_inv=tau_inv)
    behaviour = IzhikevichSpikingBehaviour(
        p=params,
        s=IzhikevichState(
            v=torch.tensor(float(v_rest), requires_grad=True),
            u=torch.tensor(u_rest) * params.b,
        ),
    )
    if print:
        p, s = behaviour
        T1 = 20
        vs = []
        us = []
        cs = []
        time = []

        for t in np.arange(0, time_print, timestep_print):
            vs.append(s.v.item())
            us.append(s.u.item())
            time.append(t * timestep_print)

            if t > T1:
                input_current = current * torch.ones(1)
            else:
                input_current = 0 * torch.ones(1)
            _, s = izhikevich_step(input_current, s, p)
            cs.append(input_current)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_ylabel("Membrane potential (mV)")
        ax1.set_xlabel("Time (ms)")
        ax1.plot(time, vs)
        ax1.plot(time, cs)
    return behaviour


tonic_spiking_p = IzhikevichParameters(a=0.02, b=0.2, c=-65, d=6)
tonic_spiking = IzhikevichSpikingBehaviour(
    p=tonic_spiking_p,
    s=IzhikevichState(
        v=torch.tensor(-70.0, requires_grad=True),
        u=torch.tensor(-70) * tonic_spiking_p.b,
    ),
)

phasic_spiking_p = IzhikevichParameters(a=0.02, b=0.25, c=-65, d=6)
phasic_spiking = IzhikevichSpikingBehaviour(
    p=phasic_spiking_p,
    s=IzhikevichState(
        v=torch.tensor(-64.0, requires_grad=True),
        u=torch.tensor(-64) * phasic_spiking_p.b,
    ),
)

tonic_bursting_p = IzhikevichParameters(a=0.02, b=0.2, c=-50, d=2)
tonic_bursting = IzhikevichSpikingBehaviour(
    p=tonic_bursting_p,
    s=IzhikevichState(
        v=torch.tensor(-70.0, requires_grad=True),
        u=torch.tensor(-70) * tonic_bursting_p.b,
    ),
)

phasic_bursting_p = IzhikevichParameters(a=0.02, b=0.25, c=-55, d=0.05, tau_inv=200)
phasic_bursting = IzhikevichSpikingBehaviour(
    p=phasic_bursting_p,
    s=IzhikevichState(
        v=torch.tensor(-64.0, requires_grad=True),
        u=torch.tensor(-64) * phasic_bursting_p.b,
    ),
)

mixed_mode_p = IzhikevichParameters(a=0.02, b=0.2, c=-55, d=4, tau_inv=250)
mixed_mode = IzhikevichSpikingBehaviour(
    p=mixed_mode_p,
    s=IzhikevichState(
        v=torch.tensor(-70.0, requires_grad=True), u=torch.tensor(-70) * mixed_mode_p.b
    ),
)

spike_frequency_adaptation_p = IzhikevichParameters(
    a=0.01, b=0.2, c=-65, d=8, tau_inv=250
)
spike_frequency_adaptation = IzhikevichSpikingBehaviour(
    p=spike_frequency_adaptation_p,
    s=IzhikevichState(
        v=torch.tensor(-70.0, requires_grad=True),
        u=torch.tensor(-70) * spike_frequency_adaptation_p.b,
    ),
)

class_1_exc_p = IzhikevichParameters(
    a=0.02, b=-0.1, c=-55, d=6, mn=4.1, bias=108, tau_inv=250
)
class_1_exc = IzhikevichSpikingBehaviour(
    p=class_1_exc_p,
    s=IzhikevichState(
        v=torch.tensor(-60.0, requires_grad=True), u=torch.tensor(-60) * class_1_exc_p.b
    ),
)

class_2_exc_p = IzhikevichParameters(a=0.2, b=0.26, c=-65, d=0, tau_inv=250)
class_2_exc = IzhikevichSpikingBehaviour(
    p=class_2_exc_p,
    s=IzhikevichState(
        v=torch.tensor(-64.0, requires_grad=True), u=torch.tensor(-64) * class_2_exc_p.b
    ),
)

spike_latency_p = IzhikevichParameters(a=0.02, b=0.2, c=-65, d=6, tau_inv=250)
spike_latency = IzhikevichSpikingBehaviour(
    p=spike_latency_p,
    s=IzhikevichState(
        v=torch.tensor(-70.0, requires_grad=True),
        u=torch.tensor(-70) * spike_latency_p.b,
    ),
)

subthreshold_oscillation_p = IzhikevichParameters(
    a=0.05, b=0.26, c=-60, d=0, tau_inv=250
)
subthreshold_oscillation = IzhikevichSpikingBehaviour(
    p=subthreshold_oscillation_p,
    s=IzhikevichState(
        v=torch.tensor(-62.0, requires_grad=True),
        u=torch.tensor(-62) * subthreshold_oscillation_p.b,
    ),
)

resonator_p = IzhikevichParameters(a=0.1, b=0.26, c=-60, d=-1, tau_inv=250)
resonator = IzhikevichSpikingBehaviour(
    p=resonator_p,
    s=IzhikevichState(
        v=torch.tensor(-62.0, requires_grad=True), u=torch.tensor(-62) * resonator_p.b
    ),
)

integrator_p = IzhikevichParameters(
    a=0.02, b=-0.1, c=-55, d=6, mn=4.1, bias=108, tau_inv=250
)
integrator = IzhikevichSpikingBehaviour(
    p=integrator_p,
    s=IzhikevichState(
        v=torch.tensor(-60.0, requires_grad=True), u=torch.tensor(-60) * integrator_p.b
    ),
)

rebound_spike_p = IzhikevichParameters(a=0.03, b=0.25, c=-60, d=4, tau_inv=200)
rebound_spike = IzhikevichSpikingBehaviour(
    p=rebound_spike_p,
    s=IzhikevichState(
        v=torch.tensor(-64.0, requires_grad=True),
        u=torch.tensor(-64) * rebound_spike_p.b,
    ),
)

rebound_burst_p = IzhikevichParameters(a=0.03, b=0.25, c=-52, d=0, tau_inv=200)
rebound_burst = IzhikevichSpikingBehaviour(
    p=rebound_burst_p,
    s=IzhikevichState(
        v=torch.tensor(-64.0, requires_grad=True),
        u=torch.tensor(-64) * rebound_burst_p.b,
    ),
)

threshhold_variability_p = IzhikevichParameters(a=0.03, b=0.25, c=-60, d=4, tau_inv=250)
threshhold_variability = IzhikevichSpikingBehaviour(
    p=threshhold_variability_p,
    s=IzhikevichState(
        v=torch.tensor(-64.0, requires_grad=True),
        u=torch.tensor(-64) * threshhold_variability_p.b,
    ),
)

bistability_p = IzhikevichParameters(a=0.1, b=0.26, c=-60, d=0, tau_inv=250)
bistability = IzhikevichSpikingBehaviour(
    p=bistability_p,
    s=IzhikevichState(
        v=torch.tensor(-61.0, requires_grad=True), u=torch.tensor(-61) * bistability_p.b
    ),
)

dap_p = IzhikevichParameters(a=1.0, b=0.2, c=-60, d=-21, tau_inv=100)
dap = IzhikevichSpikingBehaviour(
    p=dap_p,
    s=IzhikevichState(
        v=torch.tensor(-70.0, requires_grad=True), u=torch.tensor(-70) * dap_p.b
    ),
)

accomodation_p = IzhikevichParameters(a=0.02, b=1.0, c=-55, d=4, tau_inv=500)
accomodation = IzhikevichSpikingBehaviour(
    p=accomodation_p,
    s=IzhikevichState(v=torch.tensor(-65.0, requires_grad=True), u=torch.tensor(-16)),
)

inhibition_induced_spiking_p = IzhikevichParameters(
    a=-0.02, b=-1.0, c=-60, d=8, tau_inv=250
)
inhibition_induced_spiking = IzhikevichSpikingBehaviour(
    p=inhibition_induced_spiking_p,
    s=IzhikevichState(
        v=torch.tensor(-63.8, requires_grad=True),
        u=torch.tensor(-63.8) * inhibition_induced_spiking_p.b,
    ),
)

inhibition_induced_bursting_p = IzhikevichParameters(
    a=-0.026, b=-1.0, c=-45, d=-2, tau_inv=250
)
inhibition_induced_bursting = IzhikevichSpikingBehaviour(
    p=inhibition_induced_bursting_p,
    s=IzhikevichState(
        v=torch.tensor(-63.8, requires_grad=True),
        u=torch.tensor(-63.8) * inhibition_induced_bursting_p.b,
    ),
)


def izhikevich_step(
    input_current: torch.Tensor,
    s: IzhikevichState,
    p: IzhikevichParameters,
    dt: float = 0.001,
) -> Tuple[torch.Tensor, IzhikevichState]:
    v_ = s.v + p.tau_inv * dt * (
        p.sq * s.v ** 2 + p.mn * s.v + p.bias - s.u + input_current
    )
    u_ = s.u + p.tau_inv * dt * p.a * (p.b * s.v - s.u)
    z_ = threshold(v_ - p.v_th, p.method, p.alpha)
    v_ = (1 - z_) * v_ + z_ * p.c
    u_ = (1 - z_) * u_ + z_ * (u_ + p.d)
    return z_, IzhikevichState(v_, u_)


def izhikevich_recurrent_step(
    input_current: torch.Tensor,
    s: IzhikevichRecurrentState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: IzhikevichParameters,
    dt: float = 0.001,
) -> Tuple[torch.Tensor, IzhikevichRecurrentState]:
    input_current = torch.nn.functional.linear(input_current, input_weights)
    recurrent_current = torch.nn.functional.linear(s.z, recurrent_weights)
    v_ = s.v + p.tau_inv * dt * (
        p.sq * s.v ** 2 + p.mn * s.v + p.bias - s.u + input_current + recurrent_current
    )
    u_ = s.u + p.tau_inv * dt * p.a * (p.b * s.v - s.u)
    z_ = threshold(v_ - p.v_th, p.method, p.alpha)
    v_ = (1 - z_) * v_ + z_ * p.c
    u_ = (1 - z_) * u_ + z_ * (u_ + p.d)
    return z_, IzhikevichRecurrentState(z_, v_, u_)
