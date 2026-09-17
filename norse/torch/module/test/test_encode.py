"""
Test for the stateful encoder module
"""

import torch

from ..lif import LIFRecurrent
from norse.torch.module.encode import (
    PopulationEncoder,
    ConstantCurrentLIFEncoder,
    RankOrderEncoder,
    SpikeLatencyEncoder,
    PoissonEncoder,
    PoissonEncoderStep,
    SignedPoissonEncoder,
    SignedPoissonEncoderStep,
    SpikeLatencyLIFEncoder,
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


def test_signed_poisson_encoder():
    seq_length = 100
    encoder = SignedPoissonEncoder(seq_length=seq_length)
    x = torch.randn(10, 10)
    out = encoder(x)
    assert out.shape == torch.Size([seq_length, 10, 10])


def test_signed_poisson_encoder_step():
    encoder = SignedPoissonEncoderStep()
    x = torch.randn(10, 10)
    out = encoder(x)
    assert out.shape == torch.Size([10, 10])


def test_poisson_encoder():
    seq_length = 100
    encoder = PoissonEncoder(seq_length=seq_length)
    x = torch.randn(10, 10)
    out = encoder(x)
    assert out.shape == torch.Size([seq_length, 10, 10])


def test_poisson_encoder_step():
    encoder = PoissonEncoderStep()
    x = torch.randn(10, 10)
    out = encoder(x)
    assert out.shape == torch.Size([10, 10])


def test_spike_latency_lif_encoder():
    seq_length = 100
    encoder = SpikeLatencyLIFEncoder(seq_length=seq_length)
    x = torch.randn(10, 10)
    out = encoder(x)
    assert out.shape == torch.Size([seq_length, 10, 10])


def test_spike_latency_lif_encoder_backward():
    timesteps = 20
    data = torch.randn(7, 5) + 10
    encoder = SpikeLatencyLIFEncoder(timesteps)
    layer = LIFRecurrent(5, 2)
    encoded = encoder(data)
    out, _ = layer(encoded)
    out.sum().backward()


def test_spike_latency_encode_max_spikes():
    encoder = torch.nn.Sequential(
        ConstantCurrentLIFEncoder(seq_length=128), SpikeLatencyEncoder()
    )
    spikes = encoder(1.1 * torch.ones(10))
    assert torch.sum(spikes).data == 10


def test_spike_latency_encode_backward():
    torch.autograd.set_detect_anomaly(True)
    timesteps = 20
    data = torch.randn(7, 5) + 10
    encoder = torch.nn.Sequential(
        ConstantCurrentLIFEncoder(seq_length=timesteps), SpikeLatencyEncoder()
    )
    layer = LIFRecurrent(5, 2)
    encoded = encoder(data)
    out, _ = layer(encoded)
    out = out.sum()
    out.backward()


def test_spike_latency_encode_chain():
    data = torch.randn(7, 5) + 10
    encoder = torch.nn.Sequential(ConstantCurrentLIFEncoder(2), SpikeLatencyEncoder())
    encoder(data)


def test_rank_order_encoder():
    data = torch.rand(3, 5)
    encoder = RankOrderEncoder()
    out = encoder(data)
    assert out.shape == (5, 3, 5)
    assert torch.equal(out.sum(dim=0), torch.ones(3, 5))


def test_rank_order_encoder_num_steps():
    data = torch.rand(3, 5)
    encoder = RankOrderEncoder(num_steps=2)
    out = encoder(data)
    assert out.shape == (2, 3, 5)


def test_rank_order_encoder_feeding_a_neuron():
    """
    Confirm the encoder is stateless.

    It should chain into any recurrent layer exactly like the other
    encoders in this file.
    """
    data = torch.rand(4, 6)
    net = torch.nn.Sequential(RankOrderEncoder(), LIFRecurrent(6, 3))
    out, _ = net(data)
    assert out.shape == (6, 4, 3)
