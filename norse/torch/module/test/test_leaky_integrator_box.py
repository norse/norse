import torch

from norse.torch.module.leaky_integrator_box import (
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
