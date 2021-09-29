import torch
from norse.torch.functional.decode import spike_time_decode, sum_decode


def test_sum_decode():
    x = sum_decode(torch.ones(10, 3))
    assert torch.allclose(x, torch.ones(3) * 10)

def test_decode_sparse_retain_grad():
    x = torch.tensor([[0, 1], [1, 1], [0, 0]], dtype=torch.float32)
    assert not spike_time_decode(x).requires_grad

    x = torch.tensor([[0, 1], [1, 1], [0, 0]], dtype=torch.float32, requires_grad=True)
    assert spike_time_decode(x).requires_grad

def test_spike_time_decode():
    x = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=torch.float32, requires_grad=True)
    y = torch.tensor([[0, 1, 1, 1, 2], [1, 0, 1, 2, 2]])
    out = spike_time_decode(x)
    assert torch.all(torch.eq(out, y))
    out.sum().backward()
    assert torch.all(torch.eq(x, x.grad))

def test_spike_time_decode_sparse():
    x = torch.tensor([0, 1, 0, 0, 0, 1], dtype=torch.float32, requires_grad=True)
    y = torch.tensor([[1, 5]])
    out = spike_time_decode(x.to_sparse())
    assert torch.all(torch.eq(out, y))
    out.sum().backward()
    assert torch.all(torch.eq(x, x.grad))
