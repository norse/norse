import torch
import numpy as np

from .leaky_integrator import LIState, li_feed_forward_step, li_step


def lif_step_test():
    input = torch.ones(20)
    s = LIState(v=torch.zeros(10), i=torch.zeros(10))
    input_weights = torch.tensor(np.random.randn(10, 20)).float()

    for i in range(100):
        z, s = li_step(input, s, input_weights)


def lif_feed_forward_step_test():
    input = torch.ones(10)
    s = LIState(v=torch.zeros(10), i=torch.zeros(10))

    for i in range(100):
        z, s = li_feed_forward_step(input, s)
