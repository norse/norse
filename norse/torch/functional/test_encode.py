"""
Test for the encoder module
"""
import torch
from . import encode
import numpy as np


def encode_population_test():
    data = torch.tensor([0, 0.5, 1])
    out_features = 3
    actual = encode.population_encode(data, out_features)
    expected = torch.tensor(
        [
            [1.0000, 0.8824969, 0.6065307],
            [0.8824969, 1.0000, 0.8824969],
            [0.6065307, 0.8824969, 1.0000],
        ]
    )
    np.testing.assert_almost_equal(actual.numpy(), expected.numpy())


def encode_population_batch_test():
    data = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]])
    out_features = 3
    actual = encode.population_encode(data, out_features)
    np.testing.assert_equal(actual.size(), np.array([3, 3, 3]))


def constant_current_lif_encode_test():
    data = torch.tensor([0, 0, 0, 0])
    z = encode.constant_current_lif_encode(data, 2)
    np.testing.assert_equal(np.zeros((2, 4)), z.numpy())

    data = torch.tensor([[16, 16, 16], [32, 32, 32], [64, 64, 64], [128, 128, 128]])
    z = encode.constant_current_lif_encode(data, 10)
    np.testing.assert_equal(z[-1].numpy(), np.ones((4, 3)))
