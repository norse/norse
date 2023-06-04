import torch

from norse.torch import SpatialReceptiveField2d

def test_receptive_field():
    m = SpatialReceptiveField2d(2, 2, 2, 2, 9, bias=False)
    x = torch.randn(2, 10, 10)
    assert m(x).shape == torch.Size([3 * (2 * 2 + 2), 2, 2])