import torch

from norse.torch.module.lif_ex import LIFExCell, LIFExLayer, LIFExFeedForwardCell


def test_lif_ex_cell():
    cell = LIFExCell(2, 4)
    data = torch.randn(5, 2)
    out, _ = cell(data)

    assert out.shape == (5, 4)


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
