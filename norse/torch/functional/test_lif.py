import torch

from .lif import (
    LIFState,
    LIFFeedForwardState,
    lif_step,
    lif_feed_forward_step,
    lif_current_encoder,
)


def lif_step_test():
    x = torch.ones(20)
    s = LIFState(z=torch.zeros(10), v=torch.zeros(10), i=torch.zeros(10))
    input_weights = torch.randn(10, 20).float()
    recurrent_weights = torch.randn(10, 10).float()

    for _ in range(100):
        _, s = lif_step(x, s, input_weights, recurrent_weights)


def lif_feed_forward_step_test():
    x = torch.ones(10)
    s = LIFFeedForwardState(v=torch.zeros(10), i=torch.zeros(10))

    for _ in range(100):
        _, s = lif_feed_forward_step(x, s)


def lif_current_encoder_test():
    x = torch.ones(10)
    v = torch.zeros(10)

    for _ in range(100):
        _, v = lif_current_encoder(x, v)
