import torch
import numpy as np
from norse.torch.functional import izhikevich as izk
from norse.torch.functional.izhikevich import (
    izhikevich_feed_forward_step,
)


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
        _, s = izhikevich_feed_forward_step(input_current, s, p)
        cs.append(input_current)


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
        _, s = izhikevich_feed_forward_step(input_current, s, p)
        cs.append(input_current)


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
        _, s = izhikevich_feed_forward_step(input_current, s, p)

        cs.append(input_current)


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
        _, s = izhikevich_feed_forward_step(input_current, s, p)
        cs.append(input_current)


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
        _, s = izhikevich_feed_forward_step(input_current, s, p)
        cs.append(input_current)


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
        _, s = izhikevich_feed_forward_step(input_current, s, p)

        cs.append(input_current)


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
        _, s = izhikevich_feed_forward_step(input_current, s, p)


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
        _, s = izhikevich_feed_forward_step(input_current, s, p)


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
        _, s = izhikevich_feed_forward_step(input_current, s, p)


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
        _, s = izhikevich_feed_forward_step(input_current, s, p)


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
        _, s = izhikevich_feed_forward_step(input_current, s, p)


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
        _, s = izhikevich_feed_forward_step(input_current, s, p)


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
        _, s = izhikevich_feed_forward_step(input_current, s, p)


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
        _, s = izhikevich_feed_forward_step(input_current, s, p)


def test_threshhold_variability():
    p, s = izk.threshold_variability
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
        _, s = izhikevich_feed_forward_step(input_current, s, p)


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
        _, s = izhikevich_feed_forward_step(input_current, s, p)


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
        _, s = izhikevich_feed_forward_step(input_current, s, p)


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
        _, s = izhikevich_feed_forward_step(input_current, s, p)


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
        _, s = izhikevich_feed_forward_step(input_current, s, p)


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
    assert created.p == p
    assert created.s == s
