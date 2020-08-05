import torch

from .lif_ex import (
    LIFExState,
    LIFExFeedForwardState,
    lif_ex_step,
    lif_ex_feed_forward_step,
    lif_ex_current_encoder,
)


def lif_ex_step_test():
    x = torch.ones(20)
    s = LIFExState(z=torch.zeros(10), v=torch.zeros(10), i=torch.zeros(10))
    input_weights = torch.randn(10, 20).float()
    recurrent_weights = torch.randn(10, 10).float()

    for _ in range(100):
        _, s = lif_ex_step(x, s, input_weights, recurrent_weights)


def lif_ex_feed_forward_step_test():
    x = torch.ones(10)
    s = LIFExFeedForwardState(v=torch.zeros(10), i=torch.zeros(10))

    for _ in range(100):
        _, s = lif_ex_feed_forward_step(x, s)


def lif_ex_current_encoder_test():
    x = torch.ones(10)
    v = torch.zeros(10)

    for _ in range(100):
        _, v = lif_ex_current_encoder(x, v)
