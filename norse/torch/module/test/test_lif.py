import torch

from norse.torch.module.lif import LIFCell, LIFLayer, LIFFeedForwardCell


def test_lif_cell():
    cell = LIFCell(2, 4)
    data = torch.randn(5, 2)
    out, s = cell(data)

    for x in s:
        assert x.shape == (5, 4)
    assert out.shape == (5, 4)


def test_lif_layer():
    layer = LIFLayer(2, 4)
    data = torch.randn(10, 5, 2)
    out, _ = layer(data)

    assert out.shape == (10, 5, 4)


def test_lif_cell_sequence():
    l1 = LIFCell(8, 6)
    l2 = LIFCell(6, 4)
    l3 = LIFCell(4, 1)
    z = torch.ones(10, 8)
    z, s1 = l1(z)
    z, s2 = l2(z)
    z, s3 = l3(z)
    assert s1.v.shape == (10, 6)
    assert s2.v.shape == (10, 4)
    assert s3.v.shape == (10, 1)
    assert z.shape == (10, 1)


def test_lif_feedforward_cell():
    layer = LIFFeedForwardCell()
    data = torch.randn(5, 2, 4)
    out, _ = layer(data)

    assert out.shape == (5, 2, 4)
