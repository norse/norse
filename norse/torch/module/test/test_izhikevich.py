import torch

from norse.torch.module.izhikevich import IzhikevichCell

def test_izhikevich_cell():
    cell = IzhikevichCell()
    data = torch.randn(5, 2)
    out, s = cell(data)
    
    for x in s:
        assert x.shape == (5, 2)
    assert out.shape == (5, 2)
