import torch
import norse.torch.module.lsnn as lsnn
import numpy as np

from nose.tools import raises


def test_lsnn_cell():
    cell = lsnn.LSNNCell(2, 10)
    state = cell.initial_state(5, "cpu")
    data = torch.zeros(5, 2)
    z, state = cell(data, state)
    np.testing.assert_equal(z.numpy(), np.zeros((5, 10)))


@raises(TypeError)
def test_lsnn_cell_param_fail():
    # pylint: disable=E1120
    _ = lsnn.LSNNCell()


@raises(TypeError)
def test_lsnn_state_fail():
    # pylint: disable=E1120
    cell = lsnn.LSNNCell(2, 10)
    cell.initial_state()


@raises(RuntimeError)
def test_lsnn_forward_shape_fail():
    cell = lsnn.LSNNCell(2, 10)
    state = cell.initial_state(5, "cpu")
    data = torch.zeros(10)
    cell.forward(data, state)


def test_lsnn_layer():
    layer = lsnn.LSNNLayer(lsnn.LSNNCell, 2, 10)
    state = layer.initial_state(5, "cpu")
    data = torch.zeros(2, 5, 2)
    z, _ = layer.forward(data, state)
    np.testing.assert_equal(z.detach().numpy(), np.zeros((2, 5, 10)))
