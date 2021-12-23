import torch
from norse.torch.functional.iaf import (
    IAFFeedForwardState,
    IAFParameters,
    IAFState,
    iaf_step,
    iaf_feed_forward_step,
)


def test_iaf_feed_forward_step_batch():
    x = torch.ones(2, 1)
    s = IAFFeedForwardState(v=torch.zeros(2, 1))

    z, s = iaf_feed_forward_step(x, s)
    assert z.shape == (2, 1)
    assert torch.all(torch.eq(s.v, x))


def test_iaf_feed_forward_step_backward():
    x = torch.ones(2, 1)
    s = IAFFeedForwardState(v=torch.zeros(2, 1))
    s.v.requires_grad = True

    z, s = iaf_feed_forward_step(x, s)
    z.sum().backward()
    assert s.v.grad_fn is not None


def test_iaf_step_batch():
    x = torch.ones(2, 10)
    s = IAFState(z=torch.zeros(2, 5), v=torch.zeros(2, 5))
    p = IAFParameters(v_reset=torch.zeros(2, 5))
    w_in = torch.randn(5, 10)
    w_rec = torch.randn(5, 5)

    z, s = iaf_step(x, s, w_in, w_rec, p)
    assert z.shape == (2, 5)
