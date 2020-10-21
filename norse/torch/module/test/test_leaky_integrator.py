import torch

from norse.torch.module.leaky_integrator import LICell, LIFeedForwardCell, LIState


def test_li_cell():
    cell = LICell(2, 4)
    data = torch.randn(5, 2)
    out, s = cell(data)

    for x in s:
        assert x.shape == (5, 4)
    assert out.shape == (5, 4)


def test_li_cell_state():
    cell = LICell(2, 4)
    data = torch.randn(5, 2)
    out, s = cell(data, LIState(torch.ones(5, 4), torch.ones(5, 4)))

    for x in s:
        assert x.shape == (5, 4)
    assert out.shape == (5, 4)


def test_cell_backward():
    model = LICell(12, 1)
    data = torch.ones(100, 12)
    out, _ = model(data)
    loss = out.sum()
    loss.backward()


def test_li_feedforward_cell():
    layer = LIFeedForwardCell()
    data = torch.randn(10, 2, 4)
    out, _ = layer(data)

    assert out.shape == (10, 2, 4)


def test_li_feedforward_cell_state():
    layer = LIFeedForwardCell()
    data = torch.randn(2, 4)
    out, s = layer(data, LIState(torch.ones(2, 4), torch.ones(2, 4)))

    for x in s:
        assert x.shape == (2, 4)
    assert out.shape == (2, 4)


def test_ff_backward():
    model = LIFeedForwardCell()
    data = torch.ones(10, 12, 1)
    out, _ = model(data)
    loss = out.sum()
    loss.backward()
