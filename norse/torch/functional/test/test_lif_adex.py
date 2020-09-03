import torch

from norse.torch.functional.lif_adex import (
    LIFAdExState,
    LIFAdExFeedForwardState,
    lif_adex_step,
    lif_adex_feed_forward_step,
    lif_adex_current_encoder,
)


def test_lif_adex_step():
    x = torch.ones(20)
    s = LIFAdExState(
        z=torch.zeros(10), v=torch.zeros(10), i=torch.zeros(10), a=torch.zeros(10)
    )
    input_weights = torch.randn(10, 20).float()
    recurrent_weights = torch.randn(10, 10).float()

    for _ in range(100):
        _, s = lif_adex_step(x, s, input_weights, recurrent_weights)


def test_lif_adex_feed_forward_step():
    x = torch.ones(10)
    s = LIFAdExFeedForwardState(v=torch.zeros(10), i=torch.zeros(10), a=torch.zeros(10))

    for _ in range(100):
        _, s = lif_adex_feed_forward_step(x, s)


def test_lif_adex_current_encoder():
    x = torch.ones(10)
    v = torch.zeros(10)
    a = torch.zeros(10)

    for _ in range(100):
        _, _, _ = lif_adex_current_encoder(x, v, a)
