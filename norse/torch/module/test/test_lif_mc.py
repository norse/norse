import torch

from norse.torch.functional.lif import LIFState
from norse.torch.module.lif_mc import LIFMCCell


def test_lif_mc_cell():
    cell = LIFMCCell(2, 4)
    data = torch.randn(5, 2)
    out, s = cell(data)
    assert s.v.shape == (5, 4)
    assert s.i.shape == (5, 4)
    assert s.z.shape == (5, 4)
    assert out.shape == (5, 4)


def test_lif_mc_cell_state():
    cell = LIFMCCell(2, 4)

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


def test_lif_mc_cell_backward():
    cell = LIFMCCell(2, 4)
    data = torch.randn(5, 2)
    out, _ = cell(data)
    out.sum().backward()
