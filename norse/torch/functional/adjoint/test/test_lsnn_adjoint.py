import torch
import numpy as np

from norse.torch.functional.adjoint.lsnn_adjoint import (
    LSNNState,
    LSNNFeedForwardState,
    lsnn_adjoint_step,
    lsnn_feed_forward_adjoint_step,
)


def test_lsnn_adjoint_step():
    input_tensor = torch.ones(10)
    s = LSNNState(
        z=torch.zeros(10), v=torch.zeros(10), i=torch.zeros(10), b=torch.zeros(10)
    )
    input_weights = torch.tensor(np.random.randn(10, 10)).float()
    recurrent_weights = torch.tensor(np.random.randn(10, 10)).float()

    for _ in range(100):
        _, s = lsnn_adjoint_step(input_tensor, s, input_weights, recurrent_weights)


def test_lsnn_feed_forward_adjoint_step():
    input_tensor = torch.ones(10)
    s = LSNNFeedForwardState(v=torch.zeros(10), i=torch.zeros(10), b=torch.zeros(10))

    for _ in range(100):
        _, s = lsnn_feed_forward_adjoint_step(input_tensor, s)
