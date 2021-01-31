import torch

from norse.torch.module.lsnn import (
    LSNN,
    LSNNRecurrent,
    LSNNCell,
    LSNNState,
    LSNNFeedForwardState,
    LSNNRecurrentCell,
)


def test_lsnn_cell():
    cell = LSNNCell()
    data = torch.ones(1, 2, 2)
    z, state = cell(data)
    assert torch.equal(z, torch.zeros((1, 2, 2)))
    z, state = cell(data, state)
    assert torch.equal(state.i, torch.ones((1, 2, 2)) * 1.8)


def test_lsnn_cell_backward():
    cell = LSNNCell()
    data = torch.ones(5, 2)
    z, _ = cell(data)
    z.sum().backward()


def test_lsnn_recurrent_cell_autapses():
    cell = LSNNRecurrentCell(2, 2, autapses=True)
    assert not torch.allclose(
        torch.zeros(2),
        (cell.recurrent_weights * torch.eye(*cell.recurrent_weights.shape)).sum(0),
    )
    s1 = LSNNState(
        z=torch.ones(1, 2),
        v=torch.zeros(1, 2),
        i=torch.zeros(1, 2),
        b=torch.zeros(1, 2),
    )
    z, s_full = cell(torch.zeros(1, 2), s1)
    s2 = LSNNState(
        z=torch.tensor([[0, 1]], dtype=torch.float32),
        v=torch.zeros(1, 2),
        i=torch.zeros(1, 2),
        b=torch.zeros(1, 2),
    )
    z, s_part = cell(torch.zeros(1, 2), s2)

    assert not s_full.i[0, 0] == s_part.i[0, 0]


def test_lsnn_recurrent_cell_no_autopses():
    cell = LSNNRecurrentCell(2, 2, autapses=False)
    assert (
        cell.recurrent_weights * torch.eye(*cell.recurrent_weights.shape)
    ).sum() == 0

    s1 = LSNNState(
        z=torch.ones(1, 2),
        v=torch.zeros(1, 2),
        i=torch.zeros(1, 2),
        b=torch.zeros(1, 2),
    )
    z, s_full = cell(torch.zeros(1, 2), s1)
    s2 = LSNNState(
        z=torch.tensor([[0, 1]], dtype=torch.float32),
        v=torch.zeros(1, 2),
        i=torch.zeros(1, 2),
        b=torch.zeros(1, 2),
    )
    z, s_part = cell(torch.zeros(1, 2), s2)

    assert s_full.i[0, 0] == s_part.i[0, 0]


def test_lsnn_recurrent_cell_backward():
    cell = LSNNRecurrentCell(2, 2)
    data = torch.ones(5, 2)
    z, _ = cell(data)
    z.sum().backward()


def test_lsnn():
    layer = LSNN()
    data = torch.zeros(2, 5, 2)
    z, s = layer.forward(data)
    assert torch.equal(z, torch.zeros((2, 5, 2)))
    assert isinstance(s, LSNNFeedForwardState)


def test_lsnn_backward():
    cell = LSNN()
    data = torch.ones(1, 2, 2)
    z, _ = cell(data)
    z.sum().backward()


def test_lsnn_recurrent():
    layer = LSNNRecurrent(2, 10)
    data = torch.zeros(2, 5, 2)
    z, s = layer.forward(data)
    assert torch.equal(z, torch.zeros((2, 5, 10)))
    assert isinstance(s, LSNNState)


def test_lsnn_recurrent_backward():
    cell = LSNNRecurrent(2, 4)
    data = torch.ones(1, 2, 2)
    z, _ = cell(data)
    z.sum().backward()
