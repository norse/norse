import torch

from norse.torch.functional.lif import LIFState
from norse.torch.module.lif_mc import LIFMCRecurrentCell


def test_lif_mc_cell():
    cell = LIFMCRecurrentCell(2, 4)
    data = torch.randn(5, 2)
    out, s = cell(data)
    assert s.v.shape == (5, 4)
    assert s.i.shape == (5, 4)
    assert s.z.shape == (5, 4)
    assert out.shape == (5, 4)


def test_lif_mc_cell_state():
    cell = LIFMCRecurrentCell(2, 4)

    input_tensor = torch.randn(5, 2)

    state = LIFState(
        z=torch.zeros(
            input_tensor.shape[0],
            cell.hidden_size,
        ),
        v=cell.p.v_leak
        * torch.ones(
            input_tensor.shape[0],
            cell.hidden_size,
        ),
        i=torch.zeros(
            input_tensor.shape[0],
            cell.hidden_size,
        ),
    )

    out, s = cell(input_tensor, state)
    assert s.v.shape == (5, 4)
    assert s.i.shape == (5, 4)
    assert s.z.shape == (5, 4)
    assert out.shape == (5, 4)


def test_lif_mc_cell_autapses():
    cell = LIFMCRecurrentCell(2, 2, autapses=True)
    assert not torch.allclose(
        torch.zeros(2),
        (cell.recurrent_weights * torch.eye(*cell.recurrent_weights.shape)).sum(0),
    )
    s1 = LIFState(z=torch.ones(1, 2), v=torch.zeros(1, 2), i=torch.zeros(1, 2))
    z, s_full = cell(torch.zeros(1, 2), s1)
    s2 = LIFState(
        z=torch.tensor([[0, 1]], dtype=torch.float32),
        v=torch.zeros(1, 2),
        i=torch.zeros(1, 2),
    )
    z, s_part = cell(torch.zeros(1, 2), s2)

    assert not s_full.i[0, 0] == s_part.i[0, 0]


def test_lif_mc_cell_no_autapses():
    cell = LIFMCRecurrentCell(2, 2, autapses=False)
    assert (
        cell.recurrent_weights * torch.eye(*cell.recurrent_weights.shape)
    ).sum() == 0

    s1 = LIFState(z=torch.ones(1, 2), v=torch.zeros(1, 2), i=torch.zeros(1, 2))
    z, s_full = cell(torch.zeros(1, 2), s1)
    s2 = LIFState(
        z=torch.tensor([[0, 1]], dtype=torch.float32),
        v=torch.zeros(1, 2),
        i=torch.zeros(1, 2),
    )
    z, s_part = cell(torch.zeros(1, 2), s2)

    assert s_full.i[0, 0] == s_part.i[0, 0]


def test_lif_mc_cell_backward():
    cell = LIFMCRecurrentCell(2, 4)
    data = torch.randn(5, 2)
    out, _ = cell(data)
    out.sum().backward()
