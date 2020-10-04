import torch

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


def test_lif_refrac_cell_backward():
    cell = LIFRefracCell(2, 4)
    data = torch.randn(5, 2)
    out, s = cell(data)
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


def test_lif_refrac_feedforward_backward():
    batch_size = 16
    cell = LIFRefracFeedForwardCell()
    x = torch.randn(batch_size, 20, 30)
    out, _ = cell(x)
    out.sum().backward()