import torch
import norse.torch.module.lif_adex as lif_adex
import numpy as np


def test_lif_adex_cell():
    cell = lif_adex.LIFAdExCell(2, 4)
    data = torch.randn(10, 5, 2)
    state = cell.initial_state(5, "cpu")
    out, _ = cell(data, state)

    np.testing.assert_equal(out.shape, np.array([5, 4]))


def test_lif_adex_layer():
    layer = lif_adex.LIFAdExLayer(2, 4)
    data = torch.randn(10, 5, 2)
    state = layer.initial_state(5, "cpu")
    out, _ = layer(data, state)

    np.testing.assert_equal(out.shape, np.array([10, 5, 4]))


def test_lif_adex_feedforward_cell():
    layer = lif_adex.LIFAdExFeedForwardCell((2, 4))
    data = torch.randn(5, 2, 4)
    state = layer.initial_state(5, "cpu")
    out, _ = layer(data, state)

    np.testing.assert_equal(out.shape, np.array([5, 2, 4]))
