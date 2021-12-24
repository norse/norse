import torch


from norse.torch.functional.lif import LIFState, LIFFeedForwardState
from norse.torch.functional.lif_refrac import (
    LIFRefracState,
    LIFRefracFeedForwardState,
    lif_refrac_feed_forward_step,
    lif_refrac_step,
)


def test_lif_refrac_step():
    x = torch.ones(20)
    s = LIFRefracState(
        lif=LIFState(z=torch.zeros(10), v=torch.zeros(10), i=torch.zeros(10)),
        rho=torch.zeros(10),
    )
    input_weights = torch.randn(10, 20).float()
    recurrent_weights = torch.randn(10, 10).float()

    for _ in range(100):
        _, s = lif_refrac_step(x, s, input_weights, recurrent_weights)


def test_lif_refrac_feed_forward_step():
    x = torch.ones(10)
    s = LIFRefracFeedForwardState(
        lif=LIFFeedForwardState(v=torch.zeros(10), i=torch.zeros(10)),
        rho=torch.zeros(10),
    )

    for _ in range(100):
        _, s = lif_refrac_feed_forward_step(x, s)
