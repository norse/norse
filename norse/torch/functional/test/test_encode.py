"""
Test for the encoder module
"""

import torch

from norse.torch.functional.encode import (
    population_encode,
    constant_current_lif_encode,
    rank_order_encode,
    spike_latency_lif_encode,
    spike_latency_encode,
    poisson_encode,
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


def test_encode_population_scale():
    data = torch.as_tensor([0, 0.5, 1])
    out_features = 3
    scale = data.max()
    actual = population_encode(data, out_features, scale=scale)
    expected = torch.as_tensor(
        [
            [1.0000, 0.8824969, 0.6065307],
            [0.8824969, 1.0000, 0.8824969],
            [0.6065307, 0.8824969, 1.0000],
        ]
    )
    assert torch.allclose(actual, expected)


def test_encode_population_batch():
    data = torch.as_tensor([[0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1]])
    out_features = 3
    actual = population_encode(data, out_features)
    assert actual.shape == (3, 4, 3)

    data = torch.randn(10, 2, 3)
    out_features = 8
    actual = population_encode(data, out_features)
    assert actual.shape == (10, 2, 3, 8)


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


def test_poisson_encode():
    generator0 = torch.Generator()
    generator1 = torch.Generator()

    seed0 = generator0.manual_seed(45)
    seed1 = generator1.manual_seed(1043)

    data = torch.as_tensor([0.0, 0.5, 1.0])
    seq_length = 10

    spikes_seed0 = poisson_encode(data, seq_length, generator=seed0).squeeze()

    spikes_seed1 = poisson_encode(data, seq_length, generator=seed0).squeeze()

    print("seed0 spikes:", spikes_seed0)
    print("seed1 spikes:", spikes_seed1)

    assert torch.equal(spikes_seed0, spikes_seed1) == False


def test_rank_order_encode_matches_docstring_example():
    data = torch.as_tensor([0.9, 0.1, 0.5])
    expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    assert torch.equal(rank_order_encode(data), expected)


def test_rank_order_encode_default_steps_equals_input_size():
    data = torch.rand(4, 6)
    out = rank_order_encode(data)
    assert out.shape == (6, 4, 6)


def test_rank_order_encode_exactly_one_spike_per_element():
    torch.manual_seed(0)
    data = torch.rand(3, 5)
    out = rank_order_encode(data)
    assert torch.equal(out.sum(dim=0), torch.ones(3, 5))


def test_rank_order_encode_largest_fires_first():
    torch.manual_seed(1)
    data = torch.rand(8, 10)
    out = rank_order_encode(data)
    top = data.argmax(dim=-1)
    fires_at_zero = out[0].argmax(dim=-1)
    assert torch.equal(fires_at_zero, top)


def test_rank_order_encode_smallest_fires_last():
    torch.manual_seed(2)
    data = torch.rand(8, 10)
    out = rank_order_encode(data)
    bottom = data.argmin(dim=-1)
    fires_at_last = out[-1].argmax(dim=-1)
    assert torch.equal(fires_at_last, bottom)


def test_rank_order_encode_truncates_with_num_steps():
    torch.manual_seed(3)
    data = torch.rand(4, 10)
    out = rank_order_encode(data, num_steps=3)
    assert out.shape == (3, 4, 10)
    # exactly the top-3-ranked elements per row fire; the rest never do
    assert out.sum().item() == 4 * 3
    top3 = data.topk(3, dim=-1).indices
    fired = out.sum(dim=0).nonzero()
    for row in range(4):
        fired_this_row = set(fired[fired[:, 0] == row][:, 1].tolist())
        assert fired_this_row == set(top3[row].tolist())


def test_rank_order_encode_ties_produce_no_double_firing():
    data = torch.tensor([1.0, 1.0, 0.5])
    out = rank_order_encode(data)
    # ties are broken deterministically (stable argsort); either way,
    # exactly one spike per element, none shared
    assert torch.equal(out.sum(dim=0), torch.ones(3))
