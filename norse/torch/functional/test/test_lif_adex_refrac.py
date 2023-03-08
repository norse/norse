import torch

from norse.torch.functional.lif_adex import LIFAdExState, LIFAdExFeedForwardState

from norse.torch.functional.lif_adex_refrac import (
    LIFAdExRefracState,
    LIFAdExRefracFeedForwardState,
    lif_adex_refrac_feed_forward_step,
    lif_adex_refrac_step,
)


def test_lif_adex_refrac_step():
    x = torch.ones(20)
    s = LIFAdExRefracState(
        lif_adex=LIFAdExState(
            z=torch.zeros(10), v=torch.zeros(10), i=torch.zeros(10), a=torch.zeros(10)
        ),
        rho=torch.zeros(10),
    )
    input_weights = torch.randn(10, 20).float()
    recurrent_weights = torch.randn(10, 10).float()
    for _ in range(100):
        _, s = lif_adex_refrac_step(x, s, input_weights, recurrent_weights)


def test_lif_refrac_feed_forward_step():
    x = torch.ones(10)
    s = LIFAdExRefracFeedForwardState(
        lif_adex=LIFAdExFeedForwardState(
            v=torch.zeros(10), i=torch.zeros(10), a=torch.zeros(10)
        ),
        rho=torch.zeros(10),
    )
    for _ in range(100):
        _, s = lif_adex_refrac_feed_forward_step(x, s)
