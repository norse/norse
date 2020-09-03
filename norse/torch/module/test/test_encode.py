"""
Test for the stateful encoder module
"""

import torch

from norse.torch.module.encode import (
    PopulationEncoder,
    ConstantCurrentLIFEncoder,
    SpikeLatencyEncoder,
)

# Fixes a linting error:
# pylint: disable=E1102


def test_encode_population():
    data = torch.as_tensor([0, 0.5, 1])
    out_features = 3
    actual = PopulationEncoder(out_features).forward(data)
    expected = torch.as_tensor(
        [
            [1.0000, 0.8824969, 0.6065307],
            [0.8824969, 1.0000, 0.8824969],
            [0.6065307, 0.8824969, 1.0000],
        ]
    )
    assert torch.allclose(actual, expected)


def test_constant_current_lif_encode():
    data = torch.as_tensor([0, 0, 0, 0])
    z = ConstantCurrentLIFEncoder(2).forward(data)
    assert torch.equal(z, torch.zeros((2, 4)))


def test_spike_latency_encode():
    data = torch.as_tensor([[0, 100, 100], [100, 100, 100]])
    encoder = torch.nn.Sequential(ConstantCurrentLIFEncoder(2), SpikeLatencyEncoder())
    actual = encoder(data)
    expected = torch.zeros((2, 2, 3))
    expected[0] = torch.as_tensor([[0, 1, 1], [1, 1, 1]])
    assert torch.equal(actual, expected)


def test_spike_latency_encode_max_spikes():
    encoder = torch.nn.Sequential(
        ConstantCurrentLIFEncoder(seq_length=128), SpikeLatencyEncoder()
    )
    spikes = encoder(1.1 * torch.ones(10))
    assert torch.sum(spikes).data == 10


def test_spike_latency_encode_chain():
    data = torch.randn(7, 5) + 10
    encoder = torch.nn.Sequential(ConstantCurrentLIFEncoder(2), SpikeLatencyEncoder())
    encoder(data)
