import torch

from norse.torch.functional.iaf import IAFFeedForwardState
from norse.torch.module.iaf import IAFCell, IAF


def test_iaf_cell_feed_forward_step_batch():
    x = torch.ones(2, 1)
    s = IAFFeedForwardState(v=torch.zeros(2, 1))

    z, s = IAFCell()(x, s)
    assert z.shape == (2, 1)
    assert torch.all(torch.eq(s.v, x))


def test_iaf_cell_backward():
    x = torch.ones(2, 1)

    z, s = IAFCell()(x)
    z.sum().backward()
    assert s.v.grad_fn is not None


def test_iaf_feed_forward():
    x = torch.ones(10, 1)
    s = IAFFeedForwardState(v=torch.zeros(2, 1))

    z, s = IAF()(x)
    assert z.shape == (10, 1)
    assert s.v.shape == (1,)
