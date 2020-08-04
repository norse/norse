import torch
import norse.torch.module.lif_ex as lif_ex
import numpy as np


def test_lif_ex_cell():
    cell = lif_ex.LIFExCell(2, 4)
    data = torch.randn(10, 5, 2)
    state = cell.initial_state(5, "cpu")
    out, _ = cell(data, state)

    np.testing.assert_equal(out.shape, np.array([5, 4]))


def test_lif_ex_layer():
    layer = lif_ex.LIFExLayer(2, 4)
    data = torch.randn(10, 5, 2)
    state = layer.initial_state(5, "cpu")
    out, _ = layer(data, state)

    np.testing.assert_equal(out.shape, np.array([10, 5, 4]))


def test_lif_ex_feedforward_cell():
    layer = lif_ex.LIFExFeedForwardCell((2, 4))
    data = torch.randn(5, 2, 4)
    state = layer.initial_state(5, "cpu")
    out, _ = layer(data, state)

    np.testing.assert_equal(out.shape, np.array([5, 2, 4]))
