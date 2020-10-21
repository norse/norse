import torch

from norse.torch.functional.lif import LIFState, LIFFeedForwardState
from norse.torch.functional.lif_refrac import LIFRefracState, LIFRefracFeedForwardState
from norse.torch.module.lif_refrac import LIFRefracCell, LIFRefracFeedForwardCell


def test_lif_refrac_cell():
    cell = LIFRefracCell(2, 4)
    data = torch.randn(5, 2)
    out, s = cell(data)
    assert s.rho.shape == (5, 4)
    assert s.lif.v.shape == (5, 4)
    assert s.lif.i.shape == (5, 4)
    assert s.lif.z.shape == (5, 4)
    assert out.shape == (5, 4)


def test_lif_refrac_cell_state():
    cell = LIFRefracCell(2, 4)
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
    cell = LIFRefracCell(2, 4)
    data = torch.randn(5, 2)
    out, _ = cell(data)
    out.sum().backward()


def test_lif_refrac_feedforward():
    batch_size = 16
    cell = LIFRefracFeedForwardCell()
    x = torch.randn(batch_size, 20, 30)
    out, s = cell(x)
    assert out.shape == (batch_size, 20, 30)
    assert s.lif.v.shape == (batch_size, 20, 30)
    assert s.lif.i.shape == (batch_size, 20, 30)
    assert s.rho.shape == (batch_size, 20, 30)


def test_lif_refrac_feedforward_cell():
    batch_size = 16
    cell = LIFRefracFeedForwardCell()
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


def test_lif_refrac_feedforward_backward():
    batch_size = 16
    cell = LIFRefracFeedForwardCell()
    x = torch.randn(batch_size, 20, 30)
    out, _ = cell(x)
    out.sum().backward()
