import pytest
import torch
from torch import nn

from norse.torch.module.lif import LIFCell
from norse.torch.functional.synops import (
    linear_synops_accumulator,
    conv_synops_accumulator,
    synops_accumulator,
)
from norse.torch.module.synops import SynOpsCounter


def test_linear_synops_accumulator_counts_correctly():
    # 2 spikes, fan-out of 8 → 16 synops
    spikes = torch.tensor([[1.0, 0.0, 1.0, 0.0]])  # 1 batch, 4 inputs, 2 spikes
    layer = nn.Linear(4, 8)
    result = linear_synops_accumulator(spikes, layer)
    assert result == 16


def test_linear_synops_accumulator_accumulates_state():
    spikes = torch.ones(1, 4)  # 4 spikes
    layer = nn.Linear(4, 8)
    state = linear_synops_accumulator(spikes, layer, state=10)
    assert state == 10 + 4 * 8  # 10 + 32 = 42


def test_linear_synops_accumulator_none_state():
    spikes = torch.zeros(2, 3)  # no spikes
    layer = nn.Linear(3, 5)
    result = linear_synops_accumulator(spikes, layer)
    assert result == 0


def test_conv_synops_accumulator_counts_correctly():
    # 1 batch, 2 channels, 4x4 input → 4 spikes total
    # Conv2d(2, 4, kernel_size=3) → fan-out per spike = 4 * 3 * 3 / groups(1) = 36
    spikes = torch.zeros(1, 2, 4, 4)
    spikes[0, 0, 0, 0] = 1.0  # 1 spike
    layer = nn.Conv2d(2, 4, kernel_size=3)
    result = conv_synops_accumulator(spikes, layer)
    assert result == 1 * 4 * 3 * 3  # 36


def test_conv_synops_accumulator_grouped():
    # groups=2 halves the synops per spike
    spikes = torch.ones(1, 4, 2, 2)  # 16 spikes, 4 channels
    layer = nn.Conv2d(4, 8, kernel_size=3, groups=2)
    result = conv_synops_accumulator(spikes, layer)
    expected = 16 * (8 * 3 * 3 // 2)  # fan_out per spike with groups=2
    assert result == expected


def test_synops_accumulator_dispatches_linear():
    spikes = torch.ones(1, 4)
    layer = nn.Linear(4, 6)
    result = synops_accumulator(spikes, layer)
    assert result == 4 * 6


def test_synops_accumulator_dispatches_conv():
    spikes = torch.ones(1, 2, 3, 3)  # 18 spikes
    layer = nn.Conv2d(2, 4, kernel_size=3)
    result = synops_accumulator(spikes, layer)
    assert result == 18 * 4 * 3 * 3


def test_synops_accumulator_rejects_unsupported_layer():
    spikes = torch.ones(1, 4)
    with pytest.raises(TypeError):
        synops_accumulator(spikes, nn.ReLU())


def test_synops_counter_module_linear():
    cell = LIFCell()
    counter = SynOpsCounter(nn.Linear(2, 8))
    data = torch.ones(5, 2) + 10  # drives all neurons to fire
    z, _ = cell(data)
    z_out, count = counter(z)
    assert z_out.shape == z.shape
    assert count == z.sum().item() * 8


def test_synops_counter_module_accumulates():
    cell = LIFCell()
    counter = SynOpsCounter(nn.Linear(2, 8))
    data = torch.ones(5, 2) + 10

    z1, s = cell(data)
    _, count1 = counter(z1)

    z2, s = cell(data, s)
    _, count2 = counter(z2)

    assert count2 == count1 + z2.sum().item() * 8
    assert counter.state == count2


def test_synops_counter_module_state_persists():
    layer = nn.Linear(3, 5)
    counter = SynOpsCounter(layer)
    spikes = torch.ones(2, 3)  # 6 spikes → 6*5=30 synops per step
    _, c1 = counter(spikes)
    _, c2 = counter(spikes)
    assert c1 == 30
    assert c2 == 60


def test_synops_counter_reset():
    layer = nn.Linear(3, 5)
    counter = SynOpsCounter(layer)
    spikes = torch.ones(2, 3)
    _, c1 = counter(spikes)
    assert c1 == 30
    counter.reset()
    assert counter.state is None
    _, c2 = counter(spikes)
    assert c2 == 30  # starts fresh after reset
