import torch

from norse.torch.functional.coba_lif import (
    CobaLIFState,
    CobaLIFFeedForwardState,
    coba_lif_feed_forward_step,
    coba_lif_step,
)


def test_coba_lif_step():
    x = torch.ones(20)
    s = CobaLIFState(
        z=torch.zeros(10), v=torch.zeros(10), g_e=torch.zeros(10), g_i=torch.zeros(10)
    )
    input_weights = torch.randn(10, 20).float()
    recurrent_weights = torch.randn(10, 10).float()

    for _ in range(100):
        _, s = coba_lif_step(x, s, input_weights, recurrent_weights)


def test_coba_lif_feed_forward_step():
    x = torch.ones(10)
    s = CobaLIFFeedForwardState(
        v=torch.zeros(10), g_e=torch.zeros(10), g_i=torch.zeros(10)
    )

    for _ in range(100):
        _, s = coba_lif_feed_forward_step(x, s)
