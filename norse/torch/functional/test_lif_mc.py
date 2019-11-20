import torch
import numpy as np

from .lif import LIFState, LIFFeedForwardState
from .lif_mc import lif_mc_step, lif_mc_feed_forward_step


def lif_mc_step_test():
    input = torch.ones(20)
    s = LIFState(z=torch.zeros(10), v=torch.zeros(10), i=torch.zeros(10))
    input_weights = torch.tensor(np.random.randn(10, 20)).float()
    recurrent_weights = torch.tensor(np.random.randn(10, 10)).float()
    g_coupling = torch.tensor(np.random.randn(10, 10)).float()

    for i in range(100):
        z, s = lif_mc_step(input, s, input_weights, recurrent_weights, g_coupling)


def lif_mc_feed_forward_step_test():
    input = torch.ones(10)
    s = LIFFeedForwardState(v=torch.zeros(10), i=torch.zeros(10))
    g_coupling = torch.tensor(np.random.randn(10, 10)).float()

    for i in range(100):
        z, s = lif_mc_feed_forward_step(input, s, g_coupling)
