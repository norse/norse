import torch

from norse.torch.functional.leaky_integrator_box import (
    LIBoxState,
    li_box_feed_forward_step,
    li_box_step,
)


def test_li_box_step():
    x = torch.ones(20)
    s = LIBoxState(v=torch.zeros(10))
    input_weights = torch.randn(10, 20).float()

    for _ in range(100):
        _, s = li_box_step(x, s, input_weights)


def test_li_box_feed_forward_step():
    x = torch.ones(10)
    s = LIBoxState(v=torch.zeros(10))

    for _ in range(100):
        _, s = li_box_feed_forward_step(x, s)
        _, s = li_box_feed_forward_step(x, s)
