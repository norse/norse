import torch
from norse.torch.module.conv import LConv2d


def test_lconv():
    input_tensor = torch.randn(100, 20, 16, 10, 50)
    m = LConv2d(16, 33, (3, 5))
    assert m(input_tensor).shape == torch.Size([100, 20, 33, 8, 46])
