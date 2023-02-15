import torch
from norse.torch.functional.lif_box import LIFBoxFeedForwardState
from norse.torch.module.lif_box import LIFBoxCell, LIFBox


def test_lif_box_cell_feed_forward_step_batch():
    x = torch.ones(2, 1)
    s = LIFBoxFeedForwardState(v=torch.zeros(2, 1))

    z, s = LIFBoxCell()(x, s)
    assert z.shape == (2, 1)
    assert torch.all(torch.eq(s.v, 0.1))


def test_lif_box_cell_backward():
    x = torch.ones(2, 1)

    z, s = LIFBoxCell()(x)
    z.sum().backward()
    assert s.v.grad_fn is not None


def test_lif_box_in_time():
    layer = LIFBox()
    data = torch.randn(10, 5, 2)
    out, _ = layer(data)

    assert out.shape == (10, 5, 2)
