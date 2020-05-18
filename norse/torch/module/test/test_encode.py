"""
Test for the stateful encoder module
"""

import torch
from .. import encode
import numpy as np

# Fixes a linting error:
# pylint: disable=E1102


def encode_population_test():
    data = torch.tensor([0, 0.5, 1])
    out_features = 3
    actual = encode.PopulationEncoder(out_features).forward(data)
    expected = torch.tensor(
        [
            [1.0000, 0.8824969, 0.6065307],
            [0.8824969, 1.0000, 0.8824969],
            [0.6065307, 0.8824969, 1.0000],
        ]
    )
    np.testing.assert_almost_equal(actual.numpy(), expected.numpy())


def constant_current_lif_encode_test():
    data = torch.tensor([0, 0, 0, 0])
    z = encode.ConstantCurrentLIFEncoder(2).forward(data)
    np.testing.assert_equal(np.zeros((2, 4)), z.numpy())


def spike_latency_encode_test():
    data = torch.tensor([[0, 100, 100], [100, 100, 100]])
    encoder = torch.nn.Sequential(
        encode.ConstantCurrentLIFEncoder(2), encode.SpikeLatencyEncoder()
    )
    actual = encoder(data)
    expected = np.zeros((2, 2, 3))
    expected[0] = np.array([[0, 1, 1], [1, 1, 1]])
    np.testing.assert_equal(actual.numpy(), expected)


def spike_latency_encode_max_spikes_test():
    encoder = torch.nn.Sequential(
        encode.ConstantCurrentLIFEncoder(seq_length=128), encode.SpikeLatencyEncoder()
    )
    spikes = encoder(1.1 * torch.ones(10))
    assert torch.sum(spikes).data == 10


def spike_latency_encode_chain_test():
    data = torch.randn(7, 5) + 10
    encoder = torch.nn.Sequential(
        encode.ConstantCurrentLIFEncoder(2), encode.SpikeLatencyEncoder()
    )
    encoder(data)
