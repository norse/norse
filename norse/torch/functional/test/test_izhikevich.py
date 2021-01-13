import torch
import numpy as np
import norse.torch.functional.izhikevich as izk
from norse.torch.functional.izhikevich import (
    izhikevich_step,
)

import matplotlib.pyplot as plt
import matplotlib as mpl


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


def plot_test():
    s, p = izk.tonic_spiking((1))
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

    plt.subplot(5, 4, 1)
    plt.plot(vs)
    plt.plot(cs)

    s, p = izk.phasic_spiking((1))
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

    plt.subplot(5, 4, 2)
    plt.plot(vs)
    plt.plot(cs)

    s, p = izk.tonic_bursting((1))
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

    plt.subplot(5, 4, 3)
    plt.plot(vs)
    plt.plot(cs)

    s, p = izk.phasic_bursting((1))
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

    plt.subplot(5, 4, 4)
    plt.plot(vs)
    plt.plot(cs)

    s, p = izk.mixed_mode((1))
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

    plt.subplot(5, 4, 5)
    plt.plot(vs)
    plt.plot(cs)

    s, p = izk.spike_frequency_adaptation((1))
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

    plt.subplot(5, 4, 6)
    plt.plot(vs)
    plt.plot(cs)

    s, p = izk.class_1_exc((1))
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

    plt.subplot(5, 4, 7)
    plt.plot(vs)

    s, p = izk.class_2_exc((1))
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

    plt.subplot(5, 4, 8)
    plt.plot(vs)

    s, p = izk.spike_latency((1))
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

    plt.subplot(5, 4, 9)
    plt.plot(vs)
    plt.plot(cs)

    s, p = izk.subthreshold_oscillation((1))
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

    plt.subplot(5, 4, 10)
    plt.plot(vs)
    plt.plot(cs)

    s, p = izk.resonator((1))
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

    plt.subplot(5, 4, 11)
    plt.plot(vs)
    plt.plot(cs)

    s, p = izk.integrator((1))
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

    plt.subplot(5, 4, 12)
    plt.plot(vs)
    plt.plot(cs)

    s, p = izk.rebound_spike((1))
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

    plt.subplot(5, 4, 13)
    plt.plot(vs)
    plt.plot(cs)

    s, p = izk.rebound_burst((1))
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

    plt.subplot(5, 4, 14)
    plt.plot(vs)
    plt.plot(cs)

    s, p = izk.threshhold_variability((1))
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

    plt.subplot(5, 4, 15)
    plt.plot(vs)
    plt.plot(cs)

    s, p = izk.bistability((1))
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

    plt.subplot(5, 4, 16)
    plt.plot(vs)
    plt.plot(cs)

    s, p = izk.dap((1))
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

    plt.subplot(5, 4, 17)
    plt.plot(vs)
    plt.plot(cs)

    s, p = izk.inhibition_induced_spiking((1))
    ts = np.arange(0, 350, 0.5)
    T1 = 10

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

    plt.subplot(5, 4, 18)
    plt.plot(vs)
    plt.plot(cs)

    s, p = izk.inhibition_induced_bursting((1))
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

    plt.subplot(5, 4, 19)
    plt.plot(vs)
    plt.plot(cs)

    plt.savefig("izhikevich_test.png", dpi=600)
