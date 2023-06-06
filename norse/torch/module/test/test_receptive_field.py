import torch

from norse.torch import SpatialReceptiveField2d, TemporalReceptiveField


def test_spatial_receptive_field():
    m = SpatialReceptiveField2d(2, 2, 2, 2, 9, derivatives=0, bias=False)
    x = torch.randn(3, 2, 10, 10)
    assert m(x).shape == torch.Size([3, (2 * 2 + 2), 2, 2])


def test_spatial_receptive_field_no_aggregation():
    m = SpatialReceptiveField2d(
        3, 2, 2, 2, 9, derivatives=0, bias=False, aggregate=False
    )
    assert m.conv.weight.shape == torch.Size([3, 3 * (2 * 2 + 2), 9, 9])


def test_temporal_receptive_field_1d():
    m = TemporalReceptiveField((2, 10), 3)
    y, s = m(torch.randn(3, 2, 10))
    assert y.shape == (3, 3, 2, 10)


def test_temporal_receptive_field_2d():
    m = TemporalReceptiveField((2, 10, 10), 3)
    y, s = m(torch.randn(3, 2, 10, 10))
    assert y.shape == (3, 3, 2, 10, 10)
