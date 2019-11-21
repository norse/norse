import torch
import numpy as np

from .lsnn import (
    LSNNState,
    LSNNFeedForwardState,
    lsnn_feed_forward_step,
    lsnn_step,
    ada_lif_step,
)


def lsnn_step_test():
    input = torch.ones(20)
    s = LSNNState(
        z=torch.zeros(10), v=torch.zeros(10), i=torch.zeros(10), b=torch.zeros(10)
    )
    input_weights = torch.tensor(np.random.randn(10, 20)).float()
    recurrent_weights = torch.tensor(np.random.randn(10, 10)).float()

    for i in range(100):
        z, s = lsnn_step(input, s, input_weights, recurrent_weights)

def lsnn_step_test_batch():
    input = torch.ones(16, 20)
    s = LSNNState(
        z=torch.zeros(16, 10), v=torch.zeros(16, 10), i=torch.zeros(16, 10), b=torch.zeros(16, 10)
    )
    input_weights = torch.tensor(np.random.randn(10, 20)).float()
    recurrent_weights = torch.tensor(np.random.randn(10, 10)).float()

    for i in range(100):
        z, s = lsnn_step(input, s, input_weights, recurrent_weights)

def ada_lif_step_test():
    input = torch.ones(20)
    s = LSNNState(
        z=torch.zeros(10), v=torch.zeros(10), i=torch.zeros(10), b=torch.zeros(10)
    )
    input_weights = torch.tensor(np.random.randn(10, 20)).float()
    recurrent_weights = torch.tensor(np.random.randn(10, 10)).float()

    for i in range(100):
        z, s = ada_lif_step(input, s, input_weights, recurrent_weights)


def lsnn_feed_forward_step_test():
    input = torch.ones(10)
    s = LSNNFeedForwardState(v=torch.zeros(10), i=torch.zeros(10), b=torch.zeros(10))

    for i in range(100):
        z, s = lsnn_feed_forward_step(input, s)
