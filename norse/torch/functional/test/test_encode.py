"""
Test for the encoder module
"""
import torch

from norse.torch.functional.encode import (
    population_encode,
    constant_current_lif_encode,
    spike_latency_lif_encode,
    spike_latency_encode,
)

# Fixes a linting error:
# pylint: disable=E1102


def test_encode_population():
    data = torch.as_tensor([0, 0.5, 1])
    out_features = 3
    actual = population_encode(data, out_features)
    expected = torch.as_tensor(
        [
            [1.0000, 0.8824969, 0.6065307],
            [0.8824969, 1.0000, 0.8824969],
            [0.6065307, 0.8824969, 1.0000],
        ]
    )
    assert torch.allclose(actual, expected)


def test_encode_population_augment():
    data = torch.tensor([[0, 1], [1, 0]])
    out_features = 8
    actual = population_encode(data, out_features)
    assert actual.shape == (2, 2, 8)
    assert actual[0].argmax() == 0
    assert actual[1].argmax() == 7


def test_encode_population_kernel():
    data = torch.as_tensor([0, 0.5, 1])
    out_features = 3
    distance_function = lambda x, y: x - y
    kernel = lambda x: x
    actual = population_encode(
        data,
        out_features,
        min_value=-2,
        max_value=2,
        distance_function=distance_function,
        kernel=kernel,
    )
    expected = torch.as_tensor(
        [
            [2, 0, -2],
            [2.5, 0.5, -1.5],
            [3, 1, -1],
        ]
    )
    assert torch.allclose(actual, expected)


def test_encode_population_batch():
    data = torch.as_tensor([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]])
    out_features = 3
    actual = population_encode(data, out_features)
    assert actual.shape == (3, 3, 3)


def test_constant_current_lif_encode():
    data = torch.as_tensor([0, 0, 0, 0])
    z = constant_current_lif_encode(data, 2)
    assert torch.equal(z, torch.zeros((2, 4)))

    data = torch.as_tensor([[16, 16, 16], [32, 32, 32], [64, 64, 64], [128, 128, 128]])
    z = constant_current_lif_encode(data, 10)
    assert torch.equal(z[-1], torch.ones((4, 3)))


def test_spike_latency_lif_encode():
    spikes = spike_latency_lif_encode(1.1 * torch.ones(10), seq_length=128)
    assert torch.sum(spikes).data == 10


def test_spike_latency_encode_with_batch():
    data = torch.as_tensor([[100, 100], [100, 100]])
    spikes = constant_current_lif_encode(data, 5)
    actual = spike_latency_encode(spikes)
    expected = torch.zeros((5, 2, 2))
    expected[0] = torch.as_tensor([[1, 1], [1, 1]])
    assert torch.equal(actual, expected)


def test_spike_latency_encode_without_batch():
    spikes = torch.as_tensor([[0, 1, 1, 0], [1, 1, 1, 0]])
    actual = spike_latency_encode(spikes)
    assert torch.equal(actual, torch.as_tensor([[0, 1, 1, 0], [1, 0, 0, 0]]))


def test_spike_latency_encode_without_batch_2():
    spikes = torch.as_tensor([[[0, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]])
    actual = spike_latency_encode(spikes)
    expected = torch.as_tensor([[[0, 1, 1], [1, 1, 1]], [[1, 0, 0], [0, 0, 0]]])
    assert torch.equal(actual, expected)


def test_spike_latency_encode_without_batch_3():
    spikes = torch.as_tensor(
        [
            [
                [1.0, 1.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        ]
    )
    actual = spike_latency_encode(spikes)
    expected = spikes.clone()
    expected[1] = torch.as_tensor(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 1.0],
        ]
    )
    assert torch.equal(actual, expected)
