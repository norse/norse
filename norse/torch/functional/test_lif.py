import torch
import numpy as np

from .lif import (
    LIFState,
    LIFFeedForwardState,
    lif_step,
    lif_feed_forward_step,
    lif_current_encoder,
)


def lif_step_test():
    input = torch.ones(20)
    s = LIFState(z=torch.zeros(10), v=torch.zeros(10), i=torch.zeros(10))
    input_weights = torch.tensor(np.random.randn(10, 20)).float()
    recurrent_weights = torch.tensor(np.random.randn(10, 10)).float()

    for i in range(100):
        z, s = lif_step(input, s, input_weights, recurrent_weights)


def lif_feed_forward_step_test():
    input = torch.ones(10)
    s = LIFFeedForwardState(v=torch.zeros(10), i=torch.zeros(10))

    for i in range(100):
        z, s = lif_feed_forward_step(input, s)


def lif_current_encoder_test():
    input = torch.ones(10)
    v = torch.zeros(10)

    for i in range(100):
        z, v = lif_current_encoder(input, v)
