import torch

from norse.torch.module.lif_ex import (
    LIFExCell,
    LIFExRecurrentCell,
    LIFEx,
    LIFExRecurrent,
    LIFExState,
    LIFExFeedForwardState,
)


def test_lif_ex_cell():
    cell = LIFExCell()
    data = torch.randn(5, 2)
    out, _ = cell(data)

    assert out.shape == (5, 2)


def test_lif_ex_recurrent_cell_autapses():
    cell = LIFExRecurrentCell(2, 2, autapses=True)
    assert not torch.allclose(
        torch.zeros(2),
        (cell.recurrent_weights * torch.eye(*cell.recurrent_weights.shape)).sum(0),
    )
    s1 = LIFExState(z=torch.ones(1, 2), v=torch.zeros(1, 2), i=torch.zeros(1, 2))
    z, s_full = cell(torch.zeros(1, 2), s1)
    s2 = LIFExState(
        z=torch.tensor([[0, 1]], dtype=torch.float32),
        v=torch.zeros(1, 2),
        i=torch.zeros(1, 2),
    )
    z, s_part = cell(torch.zeros(1, 2), s2)

    assert not s_full.i[0, 0] == s_part.i[0, 0]


def test_lif_ex_recurrent_cell_no_autapses():
    cell = LIFExRecurrentCell(2, 2, autapses=False)
    assert (
        cell.recurrent_weights * torch.eye(*cell.recurrent_weights.shape)
    ).sum() == 0

    s1 = LIFExState(z=torch.ones(1, 2), v=torch.zeros(1, 2), i=torch.zeros(1, 2))
    z, s_full = cell(torch.zeros(1, 2), s1)
    s2 = LIFExState(
        z=torch.tensor([[0, 1]], dtype=torch.float32),
        v=torch.zeros(1, 2),
        i=torch.zeros(1, 2),
    )
    z, s_part = cell(torch.zeros(1, 2), s2)

    assert s_full.i[0, 0] == s_part.i[0, 0]


def test_lif_ex_recurrent_state():
    layer = LIFExRecurrent(2, 4)
    input_tensor = torch.randn(10, 5, 2)

    state = LIFExState(
        z=torch.zeros(
            (input_tensor.shape[1], layer.hidden_size),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        ),
        v=layer.p.v_leak,
        i=torch.zeros(
            input_tensor.shape[1],
            layer.hidden_size,
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        ),
    )
    out, _ = layer(input_tensor, state)

    assert out.shape == (10, 5, 4)


def test_lif_ex_recurrent():
    layer = LIFExRecurrent(2, 4)
    data = torch.randn(5, 2, 2)
    out, _ = layer(data)

    assert out.shape == (5, 2, 4)


def test_lif_ex():
    layer = LIFEx()
    data = torch.randn(10, 5, 2)
    out, _ = layer(data)

    assert out.shape == (10, 5, 2)


def test_lif_ex_state():
    layer = LIFEx()
    input_tensor = torch.randn(5, 2, 4)
    state = LIFExFeedForwardState(
        v=layer.p.v_leak,
        i=torch.zeros(
            *input_tensor.shape[1:],
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        ),
    )
    out, _ = layer(input_tensor, state)

    assert out.shape == (5, 2, 4)


def test_lif_ex_cell_backward():
    layer = LIFExCell()
    data = torch.randn(2, 4)
    out, _ = layer(data)
    out.sum().backward()


def test_lif_ex_recurrent_cell_backward():
    layer = LIFExRecurrentCell(2, 4)
    data = torch.randn(2, 2)
    out, _ = layer(data)
    out.sum().backward()


def test_lif_ex_backward():
    layer = LIFEx()
    data = torch.randn(2, 4)
    out, _ = layer(data)
    out.sum().backward()


def test_lif_ex_recurrent_backward():
    layer = LIFExRecurrent(2, 4)
    data = torch.randn(5, 2, 2)
    out, _ = layer(data)
    out.sum().backward()
