import torch
import norse.torch.module.lsnn as lsnn
import numpy as np

from nose.tools import raises


def test_lsnn_cell():
    cell = lsnn.LSNNCell(2, 2)
    data = torch.ones(5, 2)
    z, state = cell(data)
    np.testing.assert_equal(z.numpy(), np.zeros((5, 2)))
    z, state = cell(data, state)
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_equal,
        state.i.detach().numpy(),
        np.zeros((5, 2)),
    )


@raises(TypeError)
def test_lsnn_cell_param_fail():
    # pylint: disable=E1120
    _ = lsnn.LSNNCell()


@raises(RuntimeError)
def test_lsnn_forward_shape_fail():
    cell = lsnn.LSNNCell(2, 10)
    data = torch.zeros(10)
    cell.forward(data)


def test_lsnn_layer():
    layer = lsnn.LSNNLayer(lsnn.LSNNCell, 2, 10)
    data = torch.zeros(2, 5, 2)
    z, s = layer.forward(data)
    np.testing.assert_equal(z.detach().numpy(), np.zeros((2, 5, 10)))
    assert isinstance(s, lsnn.LSNNState)
