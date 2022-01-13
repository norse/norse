import torch

from norse.torch.functional.lif_adex import LIFAdExState, LIFAdExFeedForwardState
from norse.torch.functional.lif_adex_refrac import (
    LIFAdExRefracState,
    LIFAdExRefracFeedForwardState,
)
from norse.torch.module.lif_adex_refrac import (
    LIFAdExRefracCell,
    LIFAdExRefracRecurrentCell,
    LIFAdExRefracRecurrent,
)


def test_lif_adex_refrac_cell():
    cell = LIFAdExRefracRecurrentCell(2, 4)
    data = torch.randn(5, 2)
    out, state = cell(data)
    assert state.rho.shape == (5, 4)
    assert state.lif_adex.v.shape == (5, 4)
    assert state.lif_adex.i.shape == (5, 4)
    assert state.lif_adex.z.shape == (5, 4)
    assert state.lif_adex.a.shape == (5, 4)
    assert out.shape == (5, 4)


def test_lif_adex_refrac_cell_state():
    cell = LIFAdExRefracRecurrentCell(2, 4)
    input_tensor = torch.randn(5, 2)

    state = LIFAdExRefracState(
        lif_adex=LIFAdExState(
            z=torch.zeros(
                input_tensor.shape[0],
                cell.hidden_size,
            ),
            v=cell.p.lif_adex.v_leak
            * torch.ones(
                input_tensor.shape[0],
                cell.hidden_size,
            ),
            i=torch.zeros(
                input_tensor.shape[0],
                cell.hidden_size,
            ),
            a=torch.zeros(
                input_tensor.shape[0],
                cell.hidden_size,
            ),
        ),
        rho=torch.zeros(
            input_tensor.shape[0],
            cell.hidden_size,
        ),
    )
    out, s = cell(input_tensor, state)
    assert s.rho.shape == (5, 4)
    assert s.lif_adex.v.shape == (5, 4)
    assert s.lif_adex.i.shape == (5, 4)
    assert s.lif_adex.z.shape == (5, 4)
    assert s.lif_adex.a.shape == (5, 4)
    assert out.shape == (5, 4)


def test_lif_adex_refrac_cell_backward():
    cell = LIFAdExRefracRecurrentCell(2, 4)
    data = torch.randn(5, 2)
    out, _ = cell(data)
    out.sum().backward()


def test_lif_adex_refrac_feedforward():
    batch_size = 16
    cell = LIFAdExRefracCell()
    x = torch.randn(batch_size, 20, 30)
    out, s = cell(x)
    assert out.shape == (batch_size, 20, 30)
    assert s.lif_adex.v.shape == (batch_size, 20, 30)
    assert s.lif_adex.a.shape == (batch_size, 20, 30)
    assert s.lif_adex.i.shape == (batch_size, 20, 30)
    assert s.rho.shape == (batch_size, 20, 30)


def test_lif_adex_refrac_feedforward_cell():
    batch_size = 16
    cell = LIFAdExRefracCell()
    input_tensor = torch.randn(batch_size, 20, 30)

    state = LIFAdExRefracFeedForwardState(
        LIFAdExFeedForwardState(
            v=cell.p.lif_adex.v_leak,
            i=torch.zeros(
                input_tensor.shape,
            ),
            a=torch.zeros(
                input_tensor.shape,
            ),
        ),
        rho=torch.zeros(
            input_tensor.shape,
        ),
    )

    out, s = cell(input_tensor, state)
    assert out.shape == (batch_size, 20, 30)
    assert s.lif_adex.v.shape == (batch_size, 20, 30)
    assert s.lif_adex.a.shape == (batch_size, 20, 30)
    assert s.lif_adex.i.shape == (batch_size, 20, 30)
    assert s.rho.shape == (batch_size, 20, 30)


def test_lif_refrac_cell_autapses():
    cell = LIFAdExRefracRecurrentCell(2, 2, autapses=True)
    assert not torch.allclose(
        torch.zeros(2),
        (cell.recurrent_weights * torch.eye(*cell.recurrent_weights.shape)).sum(0),
    )
    s1 = LIFAdExRefracState(
        rho=torch.zeros(1, 2),
        lif_adex=LIFAdExState(
            z=torch.ones(1, 2),
            v=torch.zeros(1, 2),
            i=torch.zeros(1, 2),
            a=torch.zeros(1, 2),
        ),
    )
    _, s_full = cell(torch.zeros(1, 2), s1)
    s2 = LIFAdExRefracState(
        rho=torch.zeros(1, 2),
        lif_adex=LIFAdExState(
            z=torch.tensor([[0, 1]], dtype=torch.float32),
            v=torch.zeros(1, 2),
            i=torch.zeros(1, 2),
            a=torch.zeros(1, 2),
        ),
    )
    _, s_part = cell(torch.zeros(1, 2), s2)

    assert not s_full.lif_adex.i[0, 0] == s_part.lif_adex.i[0, 0]


def test_lif_adex_refrac_cell_no_autapses():
    cell = LIFAdExRefracRecurrentCell(2, 2, autapses=False)
    assert (
        cell.recurrent_weights * torch.eye(*cell.recurrent_weights.shape)
    ).sum() == 0

    s1 = LIFAdExRefracState(
        rho=torch.zeros(1, 2),
        lif_adex=LIFAdExState(
            z=torch.ones(1, 2),
            v=torch.zeros(1, 2),
            i=torch.zeros(1, 2),
            a=torch.zeros(1, 2),
        ),
    )
    _, s_full = cell(torch.zeros(1, 2), s1)
    s2 = LIFAdExRefracState(
        rho=torch.zeros(1, 2),
        lif_adex=LIFAdExState(
            z=torch.tensor([[0, 1]], dtype=torch.float32),
            v=torch.zeros(1, 2),
            i=torch.zeros(1, 2),
            a=torch.zeros(1, 2),
        ),
    )
    _, s_part = cell(torch.zeros(1, 2), s2)

    assert s_full.lif_adex.i[0, 0] == s_part.lif_adex.i[0, 0]


def test_lif_adex_refrac_feedforward_backward():
    batch_size = 16
    cell = LIFAdExRefracCell()
    x = torch.randn(batch_size, 20, 30)
    out, _ = cell(x)
    out.sum().backward()


def test_lif_adex_refrac_recurrent_sequence():
    l1 = LIFAdExRefracRecurrent(8, 6)
    l2 = LIFAdExRefracRecurrent(6, 4)
    l3 = LIFAdExRefracRecurrent(4, 1)
    z = torch.ones(10, 1, 8)
    z, s1 = l1(z)
    z, s2 = l2(z)
    z, s3 = l3(z)
    assert s1.lif_adex.v.shape == (1, 6)
    assert s2.lif_adex.v.shape == (1, 4)
    assert s3.lif_adex.v.shape == (1, 1)
    assert s1.rho.shape == (1, 6)
    assert s2.rho.shape == (1, 4)
    assert s3.rho.shape == (1, 1)
    assert z.shape == (10, 1, 1)


def test_lif_adex_refrac_recurrent_layer_backward_iteration():
    model = LIFAdExRefracRecurrent(6, 6)
    data = torch.ones(10, 6)
    out, s = model(data)
    out, _ = model(out, s)
    loss = out.sum()
    loss.backward()


def test_lif_adex_refrac_recurrent_layer_backward():
    model = LIFAdExRefracRecurrent(6, 6)
    data = torch.ones(10, 6)
    out, _ = model(data)
    loss = out.sum()
    loss.backward()


def test_lif_adex_refrac_recurrent_layer_backward_state():
    model = LIFAdExRefracRecurrent(6, 6)
    data = torch.ones(10, 6)
    out, s = model(data)
    out, _ = model(data, s)
    loss = out.sum()
    loss.backward()
