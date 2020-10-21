import torch

from norse.torch.functional.lif import LIFState
from norse.torch.functional.lif_refrac import LIFRefracState
from norse.torch.module.lif_mc_refrac import LIFMCRefracCell


def test_lif_mc_cell():
    cell = LIFMCRefracCell(2, 4)
    data = torch.randn(5, 2)
    out, s = cell(data)
    assert s.lif.v.shape == (5, 4)
    assert s.lif.i.shape == (5, 4)
    assert s.lif.z.shape == (5, 4)
    assert out.shape == (5, 4)


def test_lif_mc_refrac_state():
    cell = LIFMCRefracCell(2, 4)
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
    assert s.lif.v.shape == (5, 4)
    assert s.lif.i.shape == (5, 4)
    assert s.lif.z.shape == (5, 4)
    assert out.shape == (5, 4)


def test_lif_mc_cell_backward():
    cell = LIFMCRefracCell(2, 4)
    data = torch.randn(5, 2)
    out, _ = cell(data)
    out.sum().backward()
