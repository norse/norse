import torch
import norse.torch.module.lif as lif
import numpy as np


def test_lif_cell():
    cell = lif.LIFCell(2, 4)
    data = torch.randn(5, 2)
    out, _ = cell(data)

    np.testing.assert_equal(out.shape, np.array([5, 4]))


def test_lif_layer():
    layer = lif.LIFLayer(2, 4)
    data = torch.randn(10, 5, 2)
    out, _ = layer(data)

    np.testing.assert_equal(out.shape, np.array([10, 5, 4]))


def test_lif_feedforward_cell():
    layer = lif.LIFFeedForwardCell((2, 4))
    data = torch.randn(5, 2, 4)
    out, _ = layer(data)

    np.testing.assert_equal(out.shape, np.array([5, 2, 4]))
