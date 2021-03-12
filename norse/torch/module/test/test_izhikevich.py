import torch

from norse.torch.module.izhikevich import IzhikevichCell
from norse.torch.functional import izhikevich

def test_izhikevich_cell():
    shape = (5,2)
    data = torch.randn(shape)
    cell = IzhikevichCell(izhikevich.tonic_spiking)
    out, s = cell(data)
    
    for x in s:
        assert x.shape == (5, 2)
    assert out.shape == (5, 2)
