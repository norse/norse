import torch
import numpy as np

from .lif import LIFState, LIFFeedForwardState
from .lif_refrac import (
    LIFRefracState,
    LIFRefracFeedForwardState,
    LIFRefracParameters,
    lif_refrac_feed_forward_step,
    lif_refrac_step,
)


def lif_refrac_step_test():
    input = torch.ones(20)
    s = LIFRefracState(
        lif=LIFState(z=torch.zeros(10), v=torch.zeros(10), i=torch.zeros(10)),
        rho=torch.zeros(10),
    )
    input_weights = torch.tensor(np.random.randn(10, 20)).float()
    recurrent_weights = torch.tensor(np.random.randn(10, 10)).float()

    for i in range(100):
        z, s = lif_refrac_step(input, s, input_weights, recurrent_weights)


def lif_refrac_feed_forward_step_test():
    input = torch.ones(10)
    s = LIFRefracFeedForwardState(
        lif=LIFFeedForwardState(v=torch.zeros(10), i=torch.zeros(10)),
        rho=torch.zeros(10),
    )

    for i in range(100):
        z, s = lif_refrac_feed_forward_step(input, s)
