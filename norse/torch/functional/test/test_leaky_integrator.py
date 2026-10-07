import pytest
import torch

from norse.torch.functional.leaky_integrator import (
    LIParameters,
    LIState,
    li_feed_forward_step,
    li_step,
)


@pytest.mark.parametrize("weighted_input", [False, True])
@pytest.mark.parametrize("integer_params", [False, True])
def test_li_float16_current_decay(weighted_input, integer_params):
    x = torch.tensor([400.0], dtype=torch.float16)
    state = LIState(torch.zeros_like(x), torch.zeros_like(x))
    p = (
        LIParameters(tau_syn_inv=torch.tensor(200), tau_mem_inv=torch.tensor(100))
        if integer_params
        else LIParameters()
    )

    if weighted_input:
        voltage, state = li_step(x, state, torch.ones(1, 1, dtype=x.dtype), p)
    else:
        voltage, state = li_feed_forward_step(x, state, p)

    torch.testing.assert_close(voltage, torch.tensor([40.0], dtype=x.dtype))
    torch.testing.assert_close(state.i, torch.tensor([320.0], dtype=x.dtype))


def test_li_step():
    x = torch.ones(20)
    s = LIState(v=torch.zeros(10), i=torch.zeros(10))
    input_weights = torch.randn(10, 20).float()

    for _ in range(100):
        _, s = li_step(x, s, input_weights)


def test_li_feed_forward_step():
    x = torch.ones(10)
    s = LIState(v=torch.zeros(10), i=torch.zeros(10))

    for _ in range(100):
        _, s = li_feed_forward_step(x, s)
        _, s = li_feed_forward_step(x, s)
