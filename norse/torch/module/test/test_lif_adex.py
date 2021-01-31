import torch

from norse.torch.module.lif_adex import (
    LIFAdExCell,
    LIFAdExRecurrentCell,
    LIFAdEx,
    LIFAdExRecurrent,
    LIFAdExState,
    LIFAdExFeedForwardState,
)


def test_lif_adex_cell():
    cell = LIFAdExCell()
    data = torch.randn(5, 2)
    out, _ = cell(data)

    assert out.shape == (5, 2)


def test_lif_adex_cell_autapses():
    cell = LIFAdExRecurrentCell(2, 2, autapses=True)
    assert not torch.allclose(
        torch.zeros(2),
        (cell.recurrent_weights * torch.eye(*cell.recurrent_weights.shape)).sum(0),
    )
    s1 = LIFAdExState(
        z=torch.ones(1, 2),
        v=torch.zeros(1, 2),
        i=torch.zeros(1, 2),
        a=torch.zeros(1, 2),
    )
    z, s_full = cell(torch.zeros(1, 2), s1)
    s2 = LIFAdExState(
        z=torch.tensor([[0, 1]], dtype=torch.float32),
        v=torch.zeros(1, 2),
        i=torch.zeros(1, 2),
        a=torch.zeros(1, 2),
    )
    z, s_part = cell(torch.zeros(1, 2), s2)

    assert not s_full.i[0, 0] == s_part.i[0, 0]


def test_lif_adex_cell_no_autapses():
    cell = LIFAdExRecurrentCell(2, 2, autapses=False)
    assert (
        cell.recurrent_weights * torch.eye(*cell.recurrent_weights.shape)
    ).sum() == 0

    s1 = LIFAdExState(
        z=torch.ones(1, 2),
        v=torch.zeros(1, 2),
        i=torch.zeros(1, 2),
        a=torch.zeros(1, 2),
    )
    z, s_full = cell(torch.zeros(1, 2), s1)
    s2 = LIFAdExState(
        z=torch.tensor([[0, 1]], dtype=torch.float32),
        v=torch.zeros(1, 2),
        i=torch.zeros(1, 2),
        a=torch.zeros(1, 2),
    )
    z, s_part = cell(torch.zeros(1, 2), s2)

    assert s_full.i[0, 0] == s_part.i[0, 0]


def test_lif_adex_feedforward_cell_state():
    cell = LIFAdExCell()
    input_tensor = torch.randn(5, 2, 4)

    state = LIFAdExFeedForwardState(
        v=cell.p.v_leak,
        i=torch.zeros(
            input_tensor.shape,
        ),
        a=torch.zeros(
            input_tensor.shape,
        ),
    )

    out, _ = cell(input_tensor, state)

    assert out.shape == (5, 2, 4)


def test_lif_adex_cell_backward():
    cell = LIFAdExCell()
    data = torch.randn(5, 2)
    out, _ = cell(data)
    out.sum().backward()


def test_lif_adex_recurrent_cell_backward():
    cell = LIFAdExRecurrentCell(2, 4)
    data = torch.randn(5, 2)
    out, _ = cell(data)
    out.sum().backward()


def test_lif_adex():
    layer = LIFAdEx()
    data = torch.randn(10, 5, 4)
    out, _ = layer(data)

    assert out.shape == (10, 5, 4)


def test_lif_adex_recurrent():
    layer = LIFAdExRecurrent(2, 4)
    data = torch.randn(2, 2)
    out, _ = layer(data)

    assert out.shape == (2, 4)


def test_lif_adex_backward():
    cell = LIFAdEx()
    data = torch.randn(5, 2, 4)
    out, _ = cell(data)
    out.sum().backward()


def test_lif_adex_recurrent_backward():
    cell = LIFAdExRecurrentCell(2, 4)
    data = torch.randn(5, 2, 2)
    out, _ = cell(data)
    out.sum().backward()
