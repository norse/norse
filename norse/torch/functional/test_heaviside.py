import torch
import numpy as np
from .heaviside import heaviside


def heaviside_test():
    np.testing.assert_equal(heaviside(torch.ones(100)).numpy(), torch.ones(100).numpy())
    np.testing.assert_equal(
        heaviside(-1.0 * torch.ones(100)).numpy(), torch.zeros(100).numpy()
    )
