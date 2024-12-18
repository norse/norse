import torch

from norse.torch.module.leaky_integrator_box import (
    LIBox,
    LIBoxCell,
    LIBoxState,
    LIBoxParameters,
)


def test_li_box_cell():
    layer = LIBoxCell()
    data = torch.randn(10, 2, 4)
    out, _ = layer(data)

    assert out.shape == (10, 2, 4)


def test_li_box_cell_state():
    layer = LIBoxCell()
    data = torch.randn(2, 4)
    out, s = layer(data, LIBoxState(torch.ones(2, 4)))

    for x in s:
        assert x.shape == (2, 4)
    assert out.shape == (2, 4)


def test_li_box_cell_backward():
    model = LIBoxCell()
    data = torch.ones(10, 12, 1)
    out, _ = model(data)
    loss = out.sum()
    loss.backward()

def test_li_box_integral():
    x = torch.ones(10, 1)
    p = LIBoxParameters(
        tau_mem_inv=torch.ones(1) * 1000,
        v_leak=torch.zeros(1),
    )
    model = LIBox(p)
    out, _ = model(x)

    assert out.shape == (10, 1)
    assert torch.all(torch.eq(out, 1))


def test_li_box_integral_numerics():
    x = torch.ones(10, 1)
    p = LIBoxParameters(
        tau_mem_inv=torch.ones(1) * 700,
        v_leak=torch.zeros(1),
    )
    model = LIBoxCell(p)
    out = []
    state = None
    for y in x:
        z, state = model(y, state)
        out.append(z)
    out = torch.stack(out)

    model2 = LIBox(p)
    out2, _ = model2(x)

    assert out.shape == (10, 1)
    assert torch.all(torch.eq(out, out2))