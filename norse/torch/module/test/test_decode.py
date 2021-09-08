import torch

from norse.torch.module.decode import SpikeTimeDecoder
from norse.torch.module.lif import LIF
from norse.torch.module.sequential import SequentialState


def test_spike_time_decode():
    x = torch.tensor([[0, 1], [1, 1], [0, 0]])
    y = torch.tensor([[0, 1, 1], [1, 0, 1]])
    assert torch.allclose(SpikeTimeDecoder()(x), y)


def test_spike_time_decode_sequential():
    model = SequentialState(LIF(), SpikeTimeDecoder())
    x, _ = model(torch.ones(10, 2))
    assert torch.all(torch.eq(x, torch.tensor([[6, 6, 9, 9], [0, 1, 0, 1]])))
