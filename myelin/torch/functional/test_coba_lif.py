import torch
import numpy as np

from .coba_lif import (
    CobaLIFState,
    CobaLIFFeedForwardState,
    coba_lif_feed_forward_step,
    coba_lif_step,
)


def coba_lif_step_test():
    input = torch.ones(20)
    s = CobaLIFState(
        z=torch.zeros(10), v=torch.zeros(10), g_e=torch.zeros(10), g_i=torch.zeros(10)
    )
    input_weights = torch.tensor(np.random.randn(10, 20)).float()
    recurrent_weights = torch.tensor(np.random.randn(10, 10)).float()

    for i in range(100):
        z, s = coba_lif_step(input, s, input_weights, recurrent_weights)


def coba_lif_feed_forward_step_test():
    input = torch.ones(10)
    s = CobaLIFFeedForwardState(
        v=torch.zeros(10), g_e=torch.zeros(10), g_i=torch.zeros(10)
    )

    for i in range(100):
        z, s = coba_lif_feed_forward_step(input, s)
