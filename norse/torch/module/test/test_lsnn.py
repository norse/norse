import torch
from pytest import raises

from norse.torch.module.lsnn import LSNNCell, LSNNLayer, LSNNState


def test_lsnn_cell():
    cell = LSNNCell(2, 2)
    data = torch.ones(5, 2)
    z, state = cell(data)
    assert torch.equal(z, torch.zeros((5, 2)))
    z, state = cell(data, state)
    with raises(AssertionError):
        assert torch.equal(state.i, torch.zeros((5, 2)))


def test_lsnn_cell_param_fail():
    # pylint: disable=E1120
    with raises(TypeError):
        _ = LSNNCell()


def test_lsnn_forward_shape_fail():
    with raises(RuntimeError):
        cell = LSNNCell(2, 10)
        data = torch.zeros(10)
        cell.forward(data)


def test_lsnn_layer():
    layer = LSNNLayer(LSNNCell, 2, 10)
    data = torch.zeros(2, 5, 2)
    z, s = layer.forward(data)
    assert torch.equal(z, torch.zeros((2, 5, 10)))
    assert isinstance(s, LSNNState)
