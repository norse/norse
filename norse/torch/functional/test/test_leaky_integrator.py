import torch

from norse.torch.functional.leaky_integrator import (
    LIState,
    li_feed_forward_step,
    li_step,
)


def test_li_step():
    x = torch.ones(20)
    s = LIState(v=torch.zeros(10), i=torch.zeros(10))
    input_weights = torch.randn(10, 20).float()

    for _ in range(100):
        _, s = li_step(x, s, input_weights)


def test_li_feed_forward_step():
    x = torch.ones(10)
    s = LIState(v=torch.zeros(10), i=torch.zeros(10))

    for _ in range(100):
        _, s = li_feed_forward_step(x, s)
        _, s = li_feed_forward_step(x, s)
