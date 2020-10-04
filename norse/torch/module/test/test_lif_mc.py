import torch

from norse.torch.module.lif_mc import LIFMCCell


def test_lif_mc_cell():
    cell = LIFMCCell(2, 4)
    data = torch.randn(5, 2)
    out, s = cell(data)
    assert s.v.shape == (5, 4)
    assert s.i.shape == (5, 4)
    assert s.z.shape == (5, 4)
    assert out.shape == (5, 4)
