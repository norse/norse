import torch

from norse.torch.functional.lsnn import (
    LSNNState,
    LSNNFeedForwardState,
    lsnn_feed_forward_step,
    lsnn_step,
    ada_lif_step,
)


def test_lsnn_step():
    x = torch.ones(20)
    s = LSNNState(
        z=torch.zeros(10), v=torch.zeros(10), i=torch.zeros(10), b=torch.zeros(10)
    )
    input_weights = torch.randn(10, 20).float()
    recurrent_weights = torch.randn(10, 10).float()

    for _ in range(100):
        _, s = lsnn_step(x, s, input_weights, recurrent_weights)


def test_lsnn_step_batch():
    x = torch.ones(16, 20)
    s = LSNNState(
        z=torch.zeros(16, 10),
        v=torch.zeros(16, 10),
        i=torch.zeros(16, 10),
        b=torch.zeros(16, 10),
    )
    input_weights = torch.randn(10, 20).float()
    recurrent_weights = torch.randn(10, 10).float()

    for _ in range(100):
        _, s = lsnn_step(x, s, input_weights, recurrent_weights)


def test_ada_lif_step():
    x = torch.ones(20)
    s = LSNNState(
        z=torch.zeros(10), v=torch.zeros(10), i=torch.zeros(10), b=torch.zeros(10)
    )
    input_weights = torch.randn(10, 20).float()
    recurrent_weights = torch.randn(10, 10).float()

    for _ in range(100):
        _, s = ada_lif_step(x, s, input_weights, recurrent_weights)


def test_lsnn_feed_forward_step():
    x = torch.ones(10)
    s = LSNNFeedForwardState(v=torch.zeros(10), i=torch.zeros(10), b=torch.zeros(10))

    for _ in range(100):
        _, s = lsnn_feed_forward_step(x, s)
