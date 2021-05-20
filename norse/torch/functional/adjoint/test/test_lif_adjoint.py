import torch

from norse.torch.functional.adjoint.lif_adjoint import (
    LIFState,
    LIFFeedForwardState,
    lif_adjoint_step,
    lif_feed_forward_adjoint_step,
)
import numpy as np


def test_lif_adjoint_step():
    input = torch.ones(1, 10)
    s = LIFState(z=torch.zeros(10), v=torch.zeros(10), i=torch.zeros(10))
    input_weights = torch.tensor(np.random.randn(10, 10)).float()
    recurrent_weights = torch.tensor(np.random.randn(10, 10)).float()
    tgt = 8

    for i in range(100):
        z, s = lif_adjoint_step(input, s, input_weights, recurrent_weights)

    assert True

def test_lif_feed_forward_adjoint_step():
    input = torch.ones(1, 10)
    s = LIFFeedForwardState(v=torch.zeros(10), i=torch.zeros(10))
    tgt = 8    

    for i in range(100):
        z, s = lif_feed_forward_adjoint_step(input, s)
