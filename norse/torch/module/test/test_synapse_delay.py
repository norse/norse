import pytest
import torch

from norse.torch import SynapseDelay, SequentialState, LIFCell
from norse.torch.functional.synapse_delay import SynapseDelayState


def test_module_matches_functional_step():
    x = torch.zeros(12, 1)
    x[2, 0] = 1.0
    delay = SynapseDelay(delay=3.0, max_delay=8)
    state = None
    outs = []
    for t in range(12):
        o, state = delay(x[t], state)
        outs.append(o)
    out = torch.stack(outs).squeeze(-1)
    nz = (out.abs() > 1e-6).nonzero().flatten().tolist()
    assert nz == [5]


def test_learn_delay_creates_parameter():
    fixed = SynapseDelay(delay=2.0, max_delay=6, learn_delay=False)
    learnable = SynapseDelay(delay=2.0, max_delay=6, learn_delay=True)
    assert not isinstance(fixed.delay, torch.nn.Parameter)
    assert isinstance(learnable.delay, torch.nn.Parameter)
    assert learnable.delay.requires_grad


def test_default_max_delay_covers_the_initial_delay():
    d = SynapseDelay(delay=5.0)
    assert d.max_delay >= 5


def test_rejects_delay_above_max_delay():
    with pytest.raises(ValueError):
        SynapseDelay(delay=9.0, max_delay=4)


def test_rejects_negative_delay():
    with pytest.raises(ValueError):
        SynapseDelay(delay=-1.0, max_delay=4)


def test_learns_a_target_delay():
    """RED/GREEN: recover a delay of 7 steps from a wrong initial value."""
    torch.manual_seed(1)
    T = 60
    x = (torch.rand(T, 1) < 0.2).float()
    target = torch.zeros_like(x)
    k = 7
    target[k:] = x[:-k]

    def train(learn_delay):
        torch.manual_seed(2)
        delay = SynapseDelay(
            delay=1.0, max_delay=16, learn_delay=learn_delay,
            kernel="gaussian", sigma=6.0,
        )
        opt = torch.optim.Adam(delay.parameters(), lr=0.15) if learn_delay else None
        loss0 = None
        for it in range(400):
            if learn_delay:
                delay.sigma = max(0.5, 6.0 * (1 - it / 400))
            state = None
            outs = []
            for t in range(T):
                o, state = delay(x[t], state)
                outs.append(o)
            out = torch.stack(outs)
            loss = ((out - target) ** 2).mean()
            if loss0 is None:
                loss0 = loss.item()
            if learn_delay:
                opt.zero_grad()
                loss.backward()
                opt.step()
        return delay.delay.detach().item(), loss0, loss.item()

    d_learned, l0, lf = train(True)
    d_frozen, l0f, lff = train(False)

    assert abs(d_learned - 7.0) < 0.2
    assert lf < 0.1 * l0
    assert d_frozen == pytest.approx(1.0)          # RED control: no movement
    assert lff == pytest.approx(l0f, abs=1e-6)

    print(f"[SynapseDelay] learned delay {d_learned:.3f} (target 7.0), "
          f"loss {l0:.4f} -> {lf:.5f}")


def test_sequential_state_composition_matches_maintainer_proposal():
    """Exactly the API Jegp proposed in issue #224:

        model = SequentialState(LIFCell(), SynapseDelay(), ...)
    """
    torch.manual_seed(0)
    model = SequentialState(
        LIFCell(),
        SynapseDelay(delay=2.0, max_delay=6, learn_delay=True),
        LIFCell(),
    )
    x = torch.rand(4, 3)
    out, state = model(x)
    assert out.shape == (4, 3)
    out.sum().backward()
    delay_params = [p for n, p in model.named_parameters() if "delay" in n]
    assert len(delay_params) == 1
    assert delay_params[0].grad is not None

    # a second call with the returned state must not error (state threading)
    out2, state2 = model(torch.rand(4, 3), state)
    assert out2.shape == (4, 3)


def test_extra_repr_does_not_crash():
    repr(SynapseDelay(delay=torch.tensor([1.0, 2.0]), max_delay=5))
    repr(SynapseDelay(delay=1.0, kernel="gaussian", sigma=2.0))
