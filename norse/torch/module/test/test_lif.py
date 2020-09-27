import torch

from norse.torch.module.lif import LIFCell, LIFLayer, LIFFeedForwardCell


def test_lif_cell():
    cell = LIFCell(2, 4)
    data = torch.randn(5, 2)
    out, _ = cell(data)

    assert out.shape == (5, 4)


def test_lif_layer():
    layer = LIFLayer(2, 4)
    data = torch.randn(10, 5, 2)
    out, _ = layer(data)

    assert out.shape == (10, 5, 4)


def test_lif_feedforward_cell():
    layer = LIFFeedForwardCell()
    data = torch.randn(5, 2, 4)
    out, _ = layer(data)

    assert out.shape == (5, 2, 4)
