import torch

from norse.torch.functional.lif import LIFFeedForwardState, LIFState
from norse.torch.functional.lif_refrac import LIFRefracState, LIFRefracFeedForwardState
from norse.torch.functional.lif_mc_refrac import (
    lif_mc_refrac_step,
    lif_mc_refrac_feed_forward_step,
)


def test_lif_refrac_step():
    input_tensor = torch.ones(20)
    s = LIFRefracState(
        lif=LIFState(z=torch.zeros(10), v=torch.zeros(10), i=torch.zeros(10)),
        rho=torch.zeros(10),
    )
    input_weights = torch.randn(10, 20).float()
    recurrent_weights = torch.randn(10, 10).float()
    g_coupling = torch.randn(10, 10).float()

    for _ in range(100):
        _, s = lif_mc_refrac_step(
            input_tensor, s, input_weights, recurrent_weights, g_coupling
        )


def test_lif_refrac_feed_forward_step():
    input_tensor = torch.ones(10)
    s = LIFRefracFeedForwardState(
        lif=LIFFeedForwardState(v=torch.zeros(10), i=torch.zeros(10)),
        rho=torch.zeros(10),
    )
    g_coupling = torch.randn(10, 10).float()

    for _ in range(100):
        _, s = lif_mc_refrac_feed_forward_step(input_tensor, s, g_coupling)
