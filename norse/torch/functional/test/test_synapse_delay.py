import pytest
import torch

from norse.torch.functional.synapse_delay import (
    SynapseDelayState,
    synapse_delay_step,
)


def test_integer_delay_is_exact():
    x = torch.zeros(12, 1)
    x[2, 0] = 1.0
    state = None
    outs = []
    for t in range(12):
        o, state = synapse_delay_step(
            x[t], state, delay=torch.tensor(3.0), max_delay=8
        )
        outs.append(o)
    out = torch.stack(outs).squeeze(-1)
    nz = (out.abs() > 1e-6).nonzero().flatten().tolist()
    assert nz == [5]
    assert out[5] == pytest.approx(1.0)


def test_zero_delay_is_passthrough():
    torch.manual_seed(0)
    state = None
    for _ in range(5):
        x = torch.randn(3, 4)
        o, state = synapse_delay_step(
            x, state, delay=torch.tensor(0.0), max_delay=4
        )
        assert torch.allclose(o, x, atol=1e-6)


def test_fractional_delay_interpolates():
    x = torch.zeros(10, 1)
    x[1, 0] = 1.0
    state = None
    outs = []
    for t in range(10):
        o, state = synapse_delay_step(
            x[t], state, delay=torch.tensor(2.25), max_delay=6, kernel="linear"
        )
        outs.append(o)
    out = torch.stack(outs).squeeze(-1)
    assert out[3] == pytest.approx(0.75, abs=1e-5)
    assert out[4] == pytest.approx(0.25, abs=1e-5)
    assert out.sum() == pytest.approx(1.0, abs=1e-5)


def test_per_channel_delay():
    x = torch.zeros(12, 2)
    x[0, 0] = 1.0
    x[0, 1] = 1.0
    state = None
    outs = []
    for t in range(12):
        o, state = synapse_delay_step(
            x[t], state, delay=torch.tensor([1.0, 4.0]), max_delay=8
        )
        outs.append(o)
    out = torch.stack(outs)
    assert out[1, 0] == pytest.approx(1.0)
    assert out[4, 1] == pytest.approx(1.0)


def test_gradcheck_wrt_input():
    torch.manual_seed(0)
    delay = torch.tensor(2.3, dtype=torch.double, requires_grad=False)
    x = torch.randn(1, 3, dtype=torch.double, requires_grad=True)

    def f(inp):
        o, _ = synapse_delay_step(inp, None, delay, max_delay=6)
        return o

    assert torch.autograd.gradcheck(f, (x,), eps=1e-6, atol=1e-4)


def test_delay_gradient_matches_finite_difference():
    torch.manual_seed(3)
    x = torch.randn(15, 1, dtype=torch.double)

    def loss_at(d_val):
        delay = torch.tensor(d_val, dtype=torch.double, requires_grad=True)
        state = None
        total = torch.zeros((), dtype=torch.double)
        for t in range(x.shape[0]):
            o, state = synapse_delay_step(x[t], state, delay, max_delay=6)
            total = total + o.sum()
        return total, delay

    total, delay = loss_at(2.3)
    total.backward()
    g_analytic = delay.grad.item()

    h = 1e-6
    yp, _ = loss_at(2.3 + h)
    ym, _ = loss_at(2.3 - h)
    g_fd = (yp.item() - ym.item()) / (2 * h)
    assert abs(g_analytic - g_fd) < 1e-4
    assert abs(g_analytic) > 1e-8  # not a degenerate all-zero case


def test_state_lazily_reinitialises_on_shape_change():
    o1, s1 = synapse_delay_step(torch.randn(2, 3), None, torch.tensor(1.0), 4)
    # a shape change (e.g. batch size) must not crash on the stale buffer
    o2, s2 = synapse_delay_step(torch.randn(5, 3), s1, torch.tensor(1.0), 4)
    assert s2.buffer.shape == (5, 5, 3)


def test_rejects_bad_kernel():
    with pytest.raises(ValueError):
        synapse_delay_step(
            torch.randn(2, 2), None, torch.tensor(1.0), 4, kernel="cubic"
        )
