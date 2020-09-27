import torch

from norse.torch.functional.tsodyks_makram import (
    stp_step,
    TsodyksMakramState,
    TsodyksMakramParameters,
)


def example(p):
    dt = 0.001
    z = torch.zeros(1000)
    z[::100] = 1.0
    z[0:10] = 0.0

    s = TsodyksMakramState(x=1.0, u=0.0)
    i = 0.0
    xs = []
    us = []
    current = []

    for ts in range(1000):
        x, s = stp_step(z[ts], s, p)
        di = -p.tau_s_inv * i
        i = i + dt * di + x
        xs += [s.x]
        us += [s.u]
        current += [i]

    xs = torch.stack(xs)
    us = torch.stack(us)
    current = torch.stack(current)

    return xs, us, current


def test_depressing():
    p = TsodyksMakramParameters(
        tau_f_inv=1 / (50.0e-3),
        tau_s_inv=1 / (20.0e-3),
        tau_d_inv=1 / (750.0e-3),
        U=0.45,
    )
    _, _, current = example(p)
    assert torch.allclose(current[100], torch.as_tensor(p.U))
    assert current[500:1000].max() < current[0:500].max()


def test_facilitating():
    p = TsodyksMakramParameters(
        tau_f_inv=1 / (750.0e-3),
        tau_s_inv=1 / (20.0e-3),
        tau_d_inv=1 / (50.0e-3),
        U=0.15,
    )
    _, _, current = example(p)
    assert torch.allclose(current[100], torch.as_tensor(p.U))
    assert current[500:1000].max() > current[0:500].max()
