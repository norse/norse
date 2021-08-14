import torch

from norse.torch.functional.adjoint.lif_adjoint import (
    LIFState,
    LIFFeedForwardState,
    lif_adjoint_step,
    lif_feed_forward_adjoint_step,
)
import numpy as np


def test_lif_adjoint_step():
    input_tensor = torch.ones(1, 10)
    s = LIFState(z=torch.zeros(10), v=torch.zeros(10), i=torch.zeros(10))
    input_weights = torch.tensor(np.random.randn(10, 10)).float()
    recurrent_weights = torch.tensor(np.random.randn(10, 10)).float()

    for _ in range(100):
        _, s = lif_adjoint_step(input_tensor, s, input_weights, recurrent_weights)


def test_lif_feed_forward_adjoint_step():
    input_tensor = torch.ones(1, 10)
    s = LIFFeedForwardState(v=torch.zeros(10), i=torch.zeros(10))

    for _ in range(100):
        _, s = lif_feed_forward_adjoint_step(input_tensor, s)
