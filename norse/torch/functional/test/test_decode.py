import torch
from norse.torch.functional.decode import sum_decode


def test_sum_decode():
    x = sum_decode(torch.ones(10, 3))
    assert torch.allclose(x, torch.ones(3) * 10)
