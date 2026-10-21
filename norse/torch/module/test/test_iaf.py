import pytest
import torch

from norse.torch.functional.iaf import IAFFeedForwardState
from norse.torch.module.iaf import IAFCell, IAF


def test_iaf_cell_feed_forward_step_batch():
    x = torch.ones(2, 1)
    s = IAFFeedForwardState(v=torch.zeros(2, 1))

    z, s = IAFCell()(x, s)
    assert z.shape == (2, 1)
    assert torch.all(torch.eq(s.v, x))


def test_iaf_cell_backward():
    x = torch.ones(2, 1)

    z, s = IAFCell()(x)
    z.sum().backward()
    assert s.v.grad_fn is not None


def test_iaf_feed_forward():
    x = torch.ones(10, 1)
    s = IAFFeedForwardState(v=torch.zeros(2, 1))

    z, s = IAF()(x)
    assert z.shape == (10, 1)
    assert s.v.shape == (1,)


@pytest.mark.parametrize("record_states", [False, True])
@pytest.mark.parametrize("initial_voltage", [0.0, 0.1])
def test_iaf_records_each_membrane_state(record_states, initial_voltage):
    x = torch.full((4, 2, 3), 0.4)
    initial_state = IAFFeedForwardState(v=torch.full((2, 3), initial_voltage))
    expected_voltage = torch.tensor(
        [0.4 + initial_voltage, 0.8 + initial_voltage, 0.0, 0.4]
    )[:, None, None].expand_as(x)
    expected_spikes = torch.tensor([0.0, 0.0, 1.0, 0.0])[:, None, None].expand_as(x)

    spikes, state = IAF(record_states=record_states)(x, initial_state)

    torch.testing.assert_close(spikes, expected_spikes)
    assert isinstance(state, IAFFeedForwardState)
    torch.testing.assert_close(
        state.v, expected_voltage if record_states else expected_voltage[-1]
    )


def test_iaf_recorded_states_retain_gradients():
    x = torch.full((4, 2, 3), 0.4, requires_grad=True)
    _, recorded = IAF(record_states=True)(x)

    cell = IAFCell()
    state = None
    voltage = []
    for step in x:
        _, state = cell(step, state)
        voltage.append(state.v)
    expected = torch.stack(voltage)
    expected_grad = torch.autograd.grad(expected.sum(), x, retain_graph=True)[0]
    recorded.v.sum().backward()

    torch.testing.assert_close(recorded.v, expected)
    torch.testing.assert_close(x.grad, expected_grad)
