import torch

from norse.torch.module.lif_ex import LIFExCell, LIFExLayer, LIFExFeedForwardCell


def test_lif_ex_cell():
    cell = LIFExCell(2, 4)
    data = torch.randn(5, 2)
    out, _ = cell(data)

    assert out.shape == (5, 4)


def test_lif_ex_cell_backward():
    cell = LIFExCell(2, 4)
    data = torch.randn(5, 2)
    out, s = cell(data)
    out.sum().backward()
    data = torch.randn(5, 2)
    out, _ = cell(data, s)
    out.sum().backward()


def test_lif_ex_layer():
    layer = LIFExLayer(2, 4)
    data = torch.randn(10, 5, 2)
    out, _ = layer(data)

    assert out.shape == (10, 5, 4)


def test_lif_ex_feedforward_cell():
    layer = LIFExFeedForwardCell((2, 4))
    data = torch.randn(5, 2, 4)
    out, _ = layer(data)

    assert out.shape == (5, 2, 4)


def test_lif_ex_feedforward_cell_backward():
    layer = LIFExFeedForwardCell((2, 4))
    data = torch.randn(5, 2, 4)
    out, s = layer(data)
    out.sum().backward()
    out, _ = layer(data, s)
    out.sum().backward()
