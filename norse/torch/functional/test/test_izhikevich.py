import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import norse.torch.functional.izhikevich as izk
from norse.torch.functional.izhikevich import (
    izhikevich_step,
)
from norse.torch.utils.plot import plot_izhikevich


mpl.rcParams["axes.spines.left"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.bottom"] = False
mpl.rcParams["xtick.top"] = False
mpl.rcParams["xtick.bottom"] = False
mpl.rcParams["xtick.labeltop"] = False
mpl.rcParams["xtick.labelbottom"] = False
mpl.rcParams["ytick.left"] = False
mpl.rcParams["ytick.right"] = False
mpl.rcParams["ytick.labelleft"] = False
mpl.rcParams["ytick.labelright"] = False


def test_tonic_spiking():
    p, s = izk.tonic_spiking
    ts = np.arange(0, 100, 0.25)
    T1 = ts[-1] / 10
    vs = []
    us = []
    cs = []

    for t in ts:
        vs.append(s.v)
        us.append(s.u)
        if t > T1:
            input_current = 14 * torch.ones(1)
        else:
            input_current = 0 * torch.ones(1)
        _, s = izhikevich_step(input_current, s, p)
        cs.append(input_current)
    return vs, cs


def test_phasic_spiking():
    p, s = izk.phasic_spiking
    T1 = 20
    vs = []
    us = []
    cs = []

    for t in np.arange(0, 200, 0.25):
        vs.append(s.v)
        us.append(s.u)

        if t > T1:
            input_current = 0.5 * torch.ones(1)
        else:
            input_current = 0 * torch.ones(1)
        _, s = izhikevich_step(input_current, s, p)
        cs.append(input_current)
    return vs, cs


def test_tonic_bursting():
    p, s = izk.tonic_bursting
    T1 = 25
    vs = []
    us = []
    cs = []

    for t in np.arange(0, 220, 0.25):
        vs.append(s.v)
        us.append(s.u)

        if t > T1:
            input_current = 15 * torch.ones(1)
        else:
            input_current = 0 * torch.ones(1)
        _, s = izhikevich_step(input_current, s, p)

        cs.append(input_current)
    return vs, cs


def test_phasic_bursting():
    p, s = izk.phasic_bursting
    T1 = 20
    vs = []
    us = []
    cs = []
    for t in np.arange(0, 200, 0.25):
        vs.append(s.v)
        us.append(s.u)

        if t > T1:
            input_current = 0.6 * torch.ones(1)
        else:
            input_current = 0 * torch.ones(1)
        _, s = izhikevich_step(input_current, s, p)
        cs.append(input_current)
    return vs, cs


def test_mixed_mode():
    p, s = izk.mixed_mode
    ts = np.arange(0, 160, 0.25)
    T1 = ts[-1] / 10
    vs = []
    us = []
    cs = []
    for t in ts:
        vs.append(s.v)
        us.append(s.u)

        if t > T1:
            input_current = 10 * torch.ones(1)
        else:
            input_current = 0 * torch.ones(1)
        _, s = izhikevich_step(input_current, s, p)
        cs.append(input_current)
    return vs, cs


def test_spike_frequency_adaptation():
    p, s = izk.spike_frequency_adaptation
    ts = np.arange(0, 85, 0.25)
    T1 = ts[-1] / 10
    vs = []
    us = []
    cs = []
    for t in ts:
        vs.append(s.v)
        us.append(s.u)

        if t > T1:
            input_current = 30 * torch.ones(1)
        else:
            input_current = 0 * torch.ones(1)
        _, s = izhikevich_step(input_current, s, p)

        cs.append(input_current)
    return vs, cs


def test_class_1_exc():
    p, s = izk.class_1_exc
    ts = np.arange(0, 300, 0.25)
    T1 = 30
    vs = []
    us = []
    cs = []
    for t in ts:
        vs.append(s.v)
        us.append(s.u)

        if t > T1:
            input_current = (0.075 * (t - T1)) * torch.ones(1)
        else:
            input_current = 0 * torch.ones(1)

        cs.append(input_current)
        _, s = izhikevich_step(input_current, s, p)
    return vs


def test_class_2_exc():
    p, s = izk.class_2_exc
    ts = np.arange(0, 300, 0.25)
    T1 = 30
    vs = []
    us = []
    cs = []
    for t in ts:
        vs.append(s.v)
        us.append(s.u)

        if t > T1:
            input_current = (-0.5 + (0.015 * (t - T1))) * torch.ones(1)
        else:
            input_current = -0.5 * torch.ones(1)

        cs.append(input_current)
        _, s = izhikevich_step(input_current, s, p)
    return vs


def test_spike_latency():
    p, s = izk.spike_latency
    ts = np.arange(0, 100, 0.2)
    T1 = ts[-1] / 10
    vs = []
    us = []
    cs = []
    for t in ts:
        vs.append(s.v)
        us.append(s.u)

        if (t > T1) and (t < T1 + 3):
            input_current = 7.4 * torch.ones(1)
        else:
            input_current = 0 * torch.ones(1)

        cs.append(input_current)
        _, s = izhikevich_step(input_current, s, p)
    return vs, cs


def test_subthreshold_oscillation():
    p, s = izk.subthreshold_oscillation
    ts = np.arange(0, 200, 0.25)
    T1 = ts[-1] / 10
    vs = []
    us = []
    cs = []
    for t in ts:
        vs.append(s.v)
        us.append(s.u)

        if (t > T1) and (t < T1 + 5):
            input_current = 2.0 * torch.ones(1)
        else:
            input_current = 0 * torch.ones(1)

        cs.append(input_current)
        _, s = izhikevich_step(input_current, s, p)
    return vs, cs


def test_resonator():
    p, s = izk.resonator
    ts = np.arange(0, 400, 0.25)
    T1 = ts[-1] / 10
    T2 = T1 + 20
    T3 = 0.7 * ts[-1]
    T4 = T3 + 40

    vs = []
    us = []
    cs = []
    for t in ts:
        vs.append(s.v)
        us.append(s.u)

        if (
            ((t > T1) and (t < T1 + 4))
            or ((t > T2) and (t < T2 + 4))
            or ((t > T3) and (t < T3 + 4))
            or ((t > T4) and (t < T4 + 4))
        ):
            input_current = 0.65 * torch.ones(1)
        else:
            input_current = 0 * torch.ones(1)

        cs.append(input_current)
        _, s = izhikevich_step(input_current, s, p)
    return vs, cs


def test_integrator():
    p, s = izk.integrator
    ts = np.arange(0, 100, 0.25)
    T1 = ts[-1] / 11
    T2 = T1 + 5
    T3 = 0.7 * ts[-1]
    T4 = T3 + 10

    vs = []
    us = []
    cs = []
    for t in ts:
        vs.append(s.v)
        us.append(s.u)

        if (
            ((t > T1) and (t < T1 + 2))
            or ((t > T2) and (t < T2 + 2))
            or ((t > T3) and (t < T3 + 2))
            or ((t > T4) and (t < T4 + 2))
        ):
            input_current = 9 * torch.ones(1)
        else:
            input_current = 0 * torch.ones(1)

        cs.append(input_current)
        _, s = izhikevich_step(input_current, s, p)
    return vs, cs


def test_rebound_spike():
    p, s = izk.rebound_spike
    ts = np.arange(0, 200, 0.2)
    T1 = 20

    vs = []
    us = []
    cs = []
    for t in ts:
        vs.append(s.v)
        us.append(s.u)

        if (t > T1) and (t < T1 + 5):
            input_current = -15 * torch.ones(1)
        else:
            input_current = 0 * torch.ones(1)

        cs.append(input_current)
        _, s = izhikevich_step(input_current, s, p)
    return vs, cs


def test_rebound_burst():
    p, s = izk.rebound_burst
    ts = np.arange(0, 200, 0.2)
    T1 = 20

    vs = []
    us = []
    cs = []
    for t in ts:
        vs.append(s.v)
        us.append(s.u)

        if (t > T1) and (t < T1 + 5):
            input_current = -15 * torch.ones(1)
        else:
            input_current = 0 * torch.ones(1)

        cs.append(input_current)
        _, s = izhikevich_step(input_current, s, p)
    return vs, cs


def test_threshhold_variability():
    p, s = izk.threshhold_variability
    ts = np.arange(0, 100, 0.25)
    T1 = 10
    T2 = 80

    vs = []
    us = []
    cs = []
    for t in ts:
        vs.append(s.v)
        us.append(s.u)

        if ((t > T1) and (t < T1 + 5)) or ((t > T2) and (t < T2 + 5)):
            input_current = 1 * torch.ones(1)
        elif (t > T2 - 10) and (t < T2 - 10 + 5):
            input_current = -6 * torch.ones(1)
        else:
            input_current = 0 * torch.ones(1)

        cs.append(input_current)
        _, s = izhikevich_step(input_current, s, p)
    return vs, cs


def test_bistability():
    p, s = izk.bistability
    ts = np.arange(0, 300, 0.25)
    T1 = ts[-1] / 8
    T2 = 216

    vs = []
    us = []
    cs = []
    for t in ts:
        vs.append(s.v)
        us.append(s.u)

        if ((t > T1) and (t < T1 + 5)) or ((t > T2) and (t < T2 + 5)):
            input_current = 1.24 * torch.ones(1)
        else:
            input_current = 0.24 * torch.ones(1)

        cs.append(input_current)
        _, s = izhikevich_step(input_current, s, p)
    return vs, cs


def test_dap():
    p, s = izk.dap
    ts = np.arange(0, 50, 0.1)
    T1 = 10

    vs = []
    us = []
    cs = []
    for t in ts:
        vs.append(s.v)
        us.append(s.u)

        if abs(t - T1) < 1:
            input_current = 20.0 * torch.ones(1)
        else:
            input_current = 0.0 * torch.ones(1)

        cs.append(input_current)
        _, s = izhikevich_step(input_current, s, p)
    return vs, cs


def test_inhibition_induced_spiking():
    p, s = izk.inhibition_induced_spiking
    ts = np.arange(0, 350, 0.5)
    vs = []
    us = []
    cs = []
    for t in ts:
        vs.append(s.v)
        us.append(s.u)

        if t < 50 or t > 250:
            input_current = 80.0 * torch.ones(1)
        else:
            input_current = 75.0 * torch.ones(1)
        cs.append(input_current)
        _, s = izhikevich_step(input_current, s, p)
    return vs, cs


def test_inhibition_induced_bursting():
    p, s = izk.inhibition_induced_bursting
    ts = np.arange(0, 350, 0.5)

    vs = []
    us = []
    cs = []
    for t in ts:
        vs.append(s.v)
        us.append(s.u)

        if t < 50 or t > 250:
            input_current = 80.0 * torch.ones(1)
        else:
            input_current = 75.0 * torch.ones(1)
        cs.append(input_current)
        _, s = izhikevich_step(input_current, s, p)
    return vs, cs


def test_creation():
    p, s = izk.tonic_spiking
    created = izk.create_izhikevich_spiking_behavior(
        a=p.a, b=p.b, c=p.c, d=p.d, v_rest=float(s.v), u_rest=float(s.u) / p.b
    )
    assert created.p == p
    assert created.s == s


def test_creation_print():
    p, s = izk.tonic_spiking
    created = izk.create_izhikevich_spiking_behavior(
        a=p.a, b=p.b, c=p.c, d=p.d, v_rest=float(s.v), u_rest=float(s.u) / p.b
    )
    plot_izhikevich(created)
    assert created.p == p
    assert created.s == s
    plt.close()


def plot_test():
    vs_tonic, cs_tonic = test_tonic_spiking()

    plt.subplot(5, 4, 1)
    plt.plot(vs_tonic)
    plt.plot(cs_tonic)

    vs_phasic_spiking, cs_phasic_spiking = test_phasic_spiking()

    plt.subplot(5, 4, 2)
    plt.plot(vs_phasic_spiking)
    plt.plot(cs_phasic_spiking)

    vs_tonic_bursting, cs_tonic_bursting = test_tonic_bursting()

    plt.subplot(5, 4, 3)
    plt.plot(vs_tonic_bursting)
    plt.plot(cs_tonic_bursting)

    vs_phasic_bursting, cs_phasic_bursting = test_phasic_bursting()

    plt.subplot(5, 4, 4)
    plt.plot(vs_phasic_bursting)
    plt.plot(cs_phasic_bursting)

    vs_mixed_mode, cs_mixed_mode = test_mixed_mode()

    plt.subplot(5, 4, 5)
    plt.plot(vs_mixed_mode)
    plt.plot(cs_mixed_mode)

    vs_sfa, cs_sfa = test_spike_frequency_adaptation()

    plt.subplot(5, 4, 6)
    plt.plot(vs_sfa)
    plt.plot(cs_sfa)

    vs_class_1_exc = test_class_1_exc()

    plt.subplot(5, 4, 7)
    plt.plot(vs_class_1_exc)

    vs_class_2_exc = test_class_2_exc()

    plt.subplot(5, 4, 8)
    plt.plot(vs_class_2_exc)

    vs_spike_latency, cs_spike_latency = test_spike_latency()

    plt.subplot(5, 4, 9)
    plt.plot(vs_spike_latency)
    plt.plot(cs_spike_latency)

    vs_so, cs_so = test_subthreshold_oscillation()

    plt.subplot(5, 4, 10)
    plt.plot(vs_so)
    plt.plot(cs_so)

    vs_resonator, cs_resonator = test_resonator()

    plt.subplot(5, 4, 11)
    plt.plot(vs_resonator)
    plt.plot(cs_resonator)

    vs_integrator, cs_integrator = test_integrator()

    plt.subplot(5, 4, 12)
    plt.plot(vs_integrator)
    plt.plot(cs_integrator)

    vs_rebound_spike, cs_rebound_spike = test_rebound_spike()

    plt.subplot(5, 4, 13)
    plt.plot(vs_rebound_spike)
    plt.plot(cs_rebound_spike)

    vs_rebound_burst, cs_rebound_burst = test_rebound_burst()

    plt.subplot(5, 4, 14)
    plt.plot(vs_rebound_burst)
    plt.plot(cs_rebound_burst)

    vs_tv, cs_tv = test_threshhold_variability()

    plt.subplot(5, 4, 15)
    plt.plot(vs_tv)
    plt.plot(cs_tv)

    vs_bistability, cs_bistability = test_bistability()

    plt.subplot(5, 4, 16)
    plt.plot(vs_bistability)
    plt.plot(cs_bistability)

    vs_dap, cs_dap = test_dap()

    plt.subplot(5, 4, 17)
    plt.plot(vs_dap)
    plt.plot(cs_dap)

    vs_iis, cs_iis = test_inhibition_induced_spiking()

    plt.subplot(5, 4, 18)
    plt.plot(vs_iis)
    plt.plot(cs_iis)

    vs_iib, cs_iib = test_inhibition_induced_bursting()

    plt.subplot(5, 4, 19)
    plt.plot(vs_iib)
    plt.plot(cs_iib)

    plt.savefig("izhikevich_test.png", dpi=600)
