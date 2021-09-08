import torch
from norse.torch.functional.decode import spike_time_decode, sum_decode


def test_sum_decode():
    x = sum_decode(torch.ones(10, 3))
    assert torch.allclose(x, torch.ones(3) * 10)


def test_spike_time_decode():
    x = torch.tensor([[0, 1], [1, 1], [0, 0]])
    y = torch.tensor([[0, 1, 1], [1, 0, 1]])
    assert torch.all(torch.eq(spike_time_decode(x), y))


def test_spike_time_decode_sparse():
    x = torch.tensor([0, 1, 0, 0, 0, 1]).to_sparse()
    y = torch.tensor([[1, 5]])
    assert torch.all(torch.eq(spike_time_decode(x), y))
