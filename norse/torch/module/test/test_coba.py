import torch

from norse.torch.module.coba_lif import CobaLIFCell


def test_coba():
    cell = CobaLIFCell(4, 3)
    data = torch.ones(5, 4)
    spikes, state = cell(data)
    assert spikes.shape == (5, 3)
    assert state.v.shape == (5, 3)


def test_coba_backward():
    cell = CobaLIFCell(4, 3)
    data = torch.ones(5, 4)
    spikes, s = cell(data)
    spikes.sum().backward()
    spikes, _ = cell(data, s)
    spikes.sum().backward()
