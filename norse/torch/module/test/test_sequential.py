import pytest

import torch
from torch import nn
import norse.torch as norse


def test_state_sequence():
    d = torch.ones(10, 1, 20)
    l = norse.LIFRecurrent(20, 6)
    z, s = norse.SequentialState(l)(d)
    assert z.shape == (10, 1, 6)
    assert s[0].v.shape == (1, 6)


def test_state_sequence_apply_no_state():
    d = torch.ones(10, 1, 1)
    m = norse.SequentialState(nn.Linear(1, 1), norse.LIFCell())
    m(d)
    m.forward(d)


def test_state_sequence_apply_with_state():
    d = torch.ones(10, 1, 1)
    m = norse.SequentialState(nn.Linear(1, 1), norse.LIFCell())
    m(d, None)
    m.forward(d, None)

    s = [None, norse.LIFFeedForwardState(torch.tensor(0), torch.tensor(0))]
    m(d, s)
    m.forward(d, s)


def test_state_sequence_list():
    d = torch.ones(10, 1, 20)
    l1 = norse.LIFRecurrent(20, 6)
    l2 = norse.LIFRecurrent(6, 2)
    s = [None, None]
    z, s = norse.SequentialState(l1, l2)(d, s)
    assert z.shape == (10, 1, 2)
    assert s[1].v.shape == (1, 2)


def test_state_sequence_norse():
    d = torch.ones(10, 2, 10)
    l1 = norse.LIFRecurrent(10, 5)
    l2 = norse.LSNNRecurrent(5, 1)
    z, (s1, s2) = norse.SequentialState(l1, l2)(d)
    assert z.shape == (10, 2, 1)
    assert s1.v.shape == (2, 5)
    assert s2.v.shape == (2, 1)


def test_state_sequence_mix():
    d = torch.ones(10, 3, 20)
    l1 = norse.LIFRecurrent(20, 10)
    l2 = torch.nn.RNN(10, 4, 2)  # 2 layers
    l3 = norse.LSNNRecurrent(4, 1)
    state = [None, torch.randn(2, 3, 4), None]
    z, (s1, s2, s3) = norse.SequentialState(l1, l2, l3)(d, state)
    assert z.shape == (10, 3, 1)
    assert s1.v.shape == (3, 10)
    assert s2.shape == (2, 3, 4)
    assert s3.v.shape == (3, 1)


def test_state_sequence_conv():
    data = torch.ones(1, 8, 16, 4, 4)  # (timestep, minibatch, channels, x, y)
    model = norse.SequentialState(
        norse.Lift(torch.nn.Conv2d(16, 8, 3)),  # (1, 8, 8, 2, 2)
        torch.nn.Flatten(2),  # (1, 8, 32)
        norse.LSNNRecurrent(32, 6),  # (1, 8, 6)
        torch.nn.RNN(6, 4, 2),  # (1, 6, 4) with 2 recurrent layers
        norse.LIFRecurrent(4, 1),  # (1, 4, 1)
    )
    model(data)


def test_backprop_through_time_works():
    model = norse.SequentialState(
        norse.LSNNRecurrent(1, 2),
        norse.LSNNRecurrent(2, 3),
        norse.LSNNRecurrent(3, 3),
        norse.LSNNRecurrent(3, 2),
        nn.Flatten(),
        norse.Lift(torch.nn.Linear(4, 2)),
        norse.Lift(torch.nn.Linear(2, 1)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_func = nn.MSELoss()
    data = torch.ones(3, 1, 2, 1)
    target = torch.ones(1, 1)
    state = None
    optimizer.zero_grad()  # clear gradients for this training step
    for x in data:
        out, state = model(x, state)
    loss = loss_func(out, target)
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()


def test_sequential_forward_state_hook():
    data = torch.ones(10, 2, 8, 1)
    model = norse.SequentialState(
        torch.nn.Linear(1, 10, bias=False),
        norse.LIFRecurrent(10, 10),
        norse.LIF(),
        norse.LIF(),
        norse.LIF(),
    )
    spikes = []
    states = []

    def forward_state_hook(mod, inp, output):
        spikes.append(inp[0])
        states.append(inp[1])

    model.register_forward_state_hooks(forward_state_hook)
    _, s = model(data)
    spikes = torch.stack(spikes).detach().cpu()
    assert spikes.shape == (4, 10, 2, 8, 10)
    assert spikes.max() > 0

    assert len(states) == 4
    assert states[0] is None
    spikes = []
    states = []
    model(data, s)
    assert hasattr(states[0], "v")  # isinstance is broken for namedtuple
    model.remove_forward_state_hooks()
    assert len(model.forward_state_hooks) == 0
    model(data, s)
    assert len(spikes) == 4


def test_sequential_debug_hook_twice():
    def dub(mod, inp, output):
        pass

    model = norse.SequentialState(
        torch.nn.Linear(1, 10, bias=False),
        norse.LIFRecurrent(10, 10),
    )
    model.register_forward_state_hooks(dub)
    with pytest.raises(ValueError):
        model.register_forward_state_hooks(dub)
