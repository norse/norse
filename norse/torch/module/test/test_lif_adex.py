import torch

from norse.torch.module.lif_adex import (
    LIFAdExCell,
    LIFAdExLayer,
    LIFAdExFeedForwardCell,
)


def test_lif_adex_cell():
    cell = LIFAdExCell(2, 4)
    data = torch.randn(5, 2)
    out, _ = cell(data)

    assert out.shape == (5, 4)


def test_lif_adex_layer():
    layer = LIFAdExLayer(2, 4)
    data = torch.randn(10, 5, 2)
    out, _ = layer(data)

    assert out.shape == (10, 5, 4)


def test_lif_adex_feedforward_cell():
    layer = LIFAdExFeedForwardCell((2, 4))
    data = torch.randn(5, 2, 4)
    out, _ = layer(data)

    assert out.shape == (5, 2, 4)
