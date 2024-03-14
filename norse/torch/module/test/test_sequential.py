import pytest, platform

import torch
from torch import nn
import norse.torch as norse


def test_state_sequence():
    d = torch.ones(10, 1, 20)
    l = norse.LIFRecurrent(20, 6)
    z, s = norse.SequentialState(l)(d)
    assert z.shape == (10, 1, 6)
    assert s[0].v.shape == (1, 6)


@pytest.mark.skipif(
    not platform.system() == "Linux", reason="Only Linux supports torch.compile"
)
def test_state_sequence_compile():
    d = torch.ones(2, 1, 20)
    l = norse.LIFRecurrent(20, 6)
    m = norse.SequentialState(l)
    m = torch.compile(m)
    z, s = m(d)
    assert z.shape == (2, 1, 6)
    assert s[0].v.shape == (1, 6)


def test_state_sequence_apply_no_state():
    d = torch.ones(2, 1, 1)
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
    data = torch.ones(1, 2, 1, 4, 4)  # (timestep, minibatch, channels, x, y)
    model = norse.SequentialState(
        norse.Lift(torch.nn.Conv2d(1, 8, 3)),  # (1, 2, 8, 2, 2)
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


def test_recurrent_sequential_stateful():
    v = norse.SequentialState(torch.nn.Linear(1, 1, bias=False))
    m = norse.RecurrentSequential(v)
    data = torch.ones(10, 1, 1)
    actual, _ = v(v(data)[0] + v(data)[0])
    pred, _ = m(*m(data))
    assert torch.allclose(actual, pred)


def test_recurrent_sequential_stateless():
    v = torch.nn.Linear(1, 1, bias=False)
    m = norse.RecurrentSequential(v)
    data = torch.ones(10, 1, 1)
    actual = v(v(data) + v(data))
    pred, _ = m(*m(data))
    assert actual.shape == pred.shape
    assert torch.allclose(actual, pred)


def test_recurrent_sequential_output_index():
    l1 = torch.nn.Linear(1, 1, bias=False)
    l2 = torch.nn.Linear(1, 1, bias=False)
    m = norse.RecurrentSequential(l1, l2, output_modules=0)
    data = torch.ones(10, 1, 1)

    actual = l1(data)
    pred, _ = m(data)
    assert torch.allclose(actual, pred)

    actual = l1(l1(data))
    pred, _ = m(pred)
    assert torch.allclose(actual, pred)


@pytest.mark.skipif(
    not platform.system() == "Linux", reason="Only Linux supports torch.compile"
)
def test_sequential_compile():
    x = torch.ones(2, 2)
    m = norse.SequentialState(
        torch.nn.Linear(2, 1),
        norse.LIFBoxCell(),
    )
    m = torch.compile(m)
    z, s = m(x)
    _, s = m(x, s)

    z.sum().backward()
    assert s[1].v.grad_fn is not None
    assert z.shape == (2, 1)
    assert not torch.all(torch.eq(s[1].v, 0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")
def test_lif_box_compile_gpu():
    x = torch.ones(2, 1, device="cuda")
    p = norse.LIFBoxParameters(
        tau_mem_inv=torch.ones(1, device="cuda") * 1000,
        v_th=torch.ones(1, device="cuda"),
        v_leak=torch.zeros(1, device="cuda"),
        v_reset=torch.zeros(1, device="cuda"),
        alpha=torch.zeros(1, device="cuda"),
    )

    m = norse.LIFBoxCell(p).cuda()
    m = torch.compile(m, mode="reduce-overhead")
    z, s = m(x)
    _, s = m(x, s)

    z.sum().backward()
    assert s.v.grad_fn is not None
    assert z.shape == (2, 1)
    assert torch.all(torch.eq(s.v, 1))
