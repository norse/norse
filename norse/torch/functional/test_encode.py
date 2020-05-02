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
