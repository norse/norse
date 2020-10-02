import torch

from norse.torch.module.leaky_integrator import LICell, LIFeedForwardCell


def test_li_cell():
    cell = LICell(2, 4)
    data = torch.randn(5, 2)
    out, s = cell(data)

    for x in s:
        assert x.shape == (5, 4)
    assert out.shape == (5, 4)


def test_lif_feedforward_cell():
    layer = LIFeedForwardCell((3,))
    data = torch.randn(5, 3)
    out, _ = layer(data)

    assert out.shape == (5, 3)
