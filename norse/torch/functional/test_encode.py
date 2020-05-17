"""
Test for the encoder module
"""
import torch
from . import encode
import numpy as np

# Fixes a linting error:
# pylint: disable=E1102


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


def spike_latency_lif_encode_test():
    spikes = encode.spike_latency_lif_encode(1.1 * torch.ones(10), seq_length=128)
    assert torch.sum(spikes).data == 10


def spike_latency_encode_with_batch_test():
    data = torch.tensor([[100, 100], [100, 100]])
    spikes = encode.constant_current_lif_encode(data, 5)
    actual = encode.spike_latency_encode(spikes).to_dense()
    expected = np.zeros((5, 2, 2))
    for i, _ in enumerate(expected):
        expected[i] = np.array([[1, 1], [0, 0]])
    np.testing.assert_equal(actual.numpy(), expected)


def spike_latency_encode_without_batch_test():
    spikes = torch.tensor([[0, 1, 1, 0], [1, 1, 1, 0]])
    actual = encode.spike_latency_encode(spikes).to_dense()
    np.testing.assert_equal(actual.numpy(), np.array([[0, 1, 1, 0], [1, 0, 0, 0]]))


def spike_latency_encode_without_batch_2_test():
    spikes = torch.tensor([[[0, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]])
    actual = encode.spike_latency_encode(spikes).to_dense()
    expected = np.array([[[0, 1, 1], [1, 0, 0]], [[1, 1, 1], [0, 0, 0]]])
    np.testing.assert_equal(actual.numpy(), expected)
