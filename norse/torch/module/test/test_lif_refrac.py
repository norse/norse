import torch

from norse.torch.functional.lif import LIFState, LIFFeedForwardState
from norse.torch.functional.lif_refrac import LIFRefracState, LIFRefracFeedForwardState
from norse.torch.module.lif_refrac import LIFRefracCell, LIFRefracRecurrentCell


def test_lif_refrac_cell():
    cell = LIFRefracRecurrentCell(2, 4)
    data = torch.randn(5, 2)
    out, s = cell(data)
    assert s.rho.shape == (5, 4)
    assert s.lif.v.shape == (5, 4)
    assert s.lif.i.shape == (5, 4)
    assert s.lif.z.shape == (5, 4)
    assert out.shape == (5, 4)


def test_lif_refrac_cell_state():
    cell = LIFRefracRecurrentCell(2, 4)
    input_tensor = torch.randn(5, 2)

    state = LIFRefracState(
        lif=LIFState(
            z=torch.zeros(
                input_tensor.shape[0],
                cell.hidden_size,
            ),
            v=cell.p.lif.v_leak
            * torch.ones(
                input_tensor.shape[0],
                cell.hidden_size,
            ),
            i=torch.zeros(
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
    assert s.lif.v.shape == (5, 4)
    assert s.lif.i.shape == (5, 4)
    assert s.lif.z.shape == (5, 4)
    assert out.shape == (5, 4)


def test_lif_refrac_cell_backward():
    cell = LIFRefracRecurrentCell(2, 4)
    data = torch.randn(5, 2)
    out, _ = cell(data)
    out.sum().backward()


def test_lif_refrac_feedforward():
    batch_size = 16
    cell = LIFRefracCell()
    x = torch.randn(batch_size, 20, 30)
    out, s = cell(x)
    assert out.shape == (batch_size, 20, 30)
    assert s.lif.v.shape == (batch_size, 20, 30)
    assert s.lif.i.shape == (batch_size, 20, 30)
    assert s.rho.shape == (batch_size, 20, 30)


def test_lif_refrac_feedforward_cell():
    batch_size = 16
    cell = LIFRefracCell()
    input_tensor = torch.randn(batch_size, 20, 30)

    state = LIFRefracFeedForwardState(
        LIFFeedForwardState(
            v=cell.p.lif.v_leak,
            i=torch.zeros(
                input_tensor.shape,
            ),
        ),
        rho=torch.zeros(
            input_tensor.shape,
        ),
    )

    out, s = cell(input_tensor, state)
    assert out.shape == (batch_size, 20, 30)
    assert s.lif.v.shape == (batch_size, 20, 30)
    assert s.lif.i.shape == (batch_size, 20, 30)
    assert s.rho.shape == (batch_size, 20, 30)


def test_lif_refrac_cell_autapses():
    cell = LIFRefracRecurrentCell(2, 2, autapses=True)
    assert not torch.allclose(
        torch.zeros(2),
        (cell.recurrent_weights * torch.eye(*cell.recurrent_weights.shape)).sum(0),
    )
    s1 = LIFRefracState(
        rho=torch.zeros(1, 2),
        lif=LIFState(z=torch.ones(1, 2), v=torch.zeros(1, 2), i=torch.zeros(1, 2)),
    )
    z, s_full = cell(torch.zeros(1, 2), s1)
    s2 = LIFRefracState(
        rho=torch.zeros(1, 2),
        lif=LIFState(
            z=torch.tensor([[0, 1]], dtype=torch.float32),
            v=torch.zeros(1, 2),
            i=torch.zeros(1, 2),
        ),
    )
    z, s_part = cell(torch.zeros(1, 2), s2)

    assert not s_full.lif.i[0, 0] == s_part.lif.i[0, 0]


def test_lif_refrac_cell_no_autapses():
    cell = LIFRefracRecurrentCell(2, 2, autapses=False)
    assert (
        cell.recurrent_weights * torch.eye(*cell.recurrent_weights.shape)
    ).sum() == 0

    s1 = LIFRefracState(
        rho=torch.zeros(1, 2),
        lif=LIFState(z=torch.ones(1, 2), v=torch.zeros(1, 2), i=torch.zeros(1, 2)),
    )
    z, s_full = cell(torch.zeros(1, 2), s1)
    s2 = LIFRefracState(
        rho=torch.zeros(1, 2),
        lif=LIFState(
            z=torch.tensor([[0, 1]], dtype=torch.float32),
            v=torch.zeros(1, 2),
            i=torch.zeros(1, 2),
        ),
    )
    z, s_part = cell(torch.zeros(1, 2), s2)

    assert s_full.lif.i[0, 0] == s_part.lif.i[0, 0]


def test_lif_refrac_feedforward_backward():
    batch_size = 16
    cell = LIFRefracCell()
    x = torch.randn(batch_size, 20, 30)
    out, _ = cell(x)
    out.sum().backward()
