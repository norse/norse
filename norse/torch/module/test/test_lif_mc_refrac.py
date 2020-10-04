import torch

from norse.torch.module.lif_mc_refrac import LIFMCRefracCell


def test_lif_mc_cell():
    cell = LIFMCRefracCell(2, 4)
    data = torch.randn(5, 2)
    out, s = cell(data)
    assert s.lif.v.shape == (5, 4)
    assert s.lif.i.shape == (5, 4)
    assert s.lif.z.shape == (5, 4)
    assert out.shape == (5, 4)

def test_lif_mc_cell_backward():
    cell = LIFMCRefracCell(2, 4)
    data = torch.randn(5, 2)
    out, s = cell(data)
    out.sum().backward()
