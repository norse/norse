import torch

from norse.torch.functional.lif_box import (
    LIFBoxFeedForwardState,
    lif_box_feed_forward_step,
)


def test_lif_feed_forward_step():
    xs = torch.tensor([1, 1, 1, 1, 1, 0, 0])
    s = LIFBoxFeedForwardState(v=10)

    results = [0.0, 0.1, 0.19, 0.271, 0.3439, 0.3095, 0.2786]

    for x, expected in zip(xs, results):
        _, s = lif_box_feed_forward_step(x, s)
        assert torch.allclose(torch.as_tensor(expected), s.v, atol=1e-4)


def test_lif_feed_forward_step_backward():
    x = torch.ones(2, 1)
    s = LIFBoxFeedForwardState(v=torch.zeros(2, 1))
    s.v.requires_grad = True

    z, s = lif_box_feed_forward_step(x, s)
    z.sum().backward()
    assert s.v.grad_fn is not None
