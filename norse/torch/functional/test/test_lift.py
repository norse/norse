import torch
import pytest

from norse.torch.functional.leaky_integrator import li_step, LIState
from norse.torch.functional.lif import (
    lif_step,
    lif_feed_forward_step,
    LIFState,
    LIFFeedForwardState,
    LIFParameters,
)
from norse.torch.functional.lift import lift


def test_lift_without_state_or_parameters():
    data = torch.ones(3, 2, 1)
    lifted = lift(lif_feed_forward_step)
    with pytest.raises(TypeError):  # No state given
        lifted(data)


def test_lift_with_state_without_parameters():
    data = torch.ones(3, 2, 1)
    lifted = lift(lif_feed_forward_step)
    z, s = lifted(
        data,
        state=LIFFeedForwardState(torch.zeros_like(data[0]), torch.zeros_like(data[0])),
        p=LIFParameters(),
    )
    assert z.shape == (3, 2, 1)
    assert s.v.shape == (2, 1)
    assert s.i.shape == (2, 1)


def test_lift_with_state_and_parameters():
    data = torch.ones(3, 2, 1)
    lifted = lift(
        lif_feed_forward_step, p=LIFParameters(v_th=torch.as_tensor(0.3), method="tanh")
    )
    z, s = lifted(
        data,
        state=LIFFeedForwardState(torch.zeros_like(data[0]), torch.zeros_like(data[0])),
        p=LIFParameters(),
    )
    assert z.shape == (3, 2, 1)
    assert s.v.shape == (2, 1)
    assert s.i.shape == (2, 1)


def test_lift_with_lift_step():
    data = torch.ones(3, 2, 1)
    lifted = lift(lif_step)
    z, s = lifted(
        data,
        state=LIFState(
            v=torch.zeros(2, 1),
            i=torch.zeros(2, 1),
            z=torch.zeros(2, 1),
        ),
        input_weights=torch.ones(1, 1),
        recurrent_weights=torch.ones(1, 1),
        p=LIFParameters(),
    )
    assert z.shape == (3, 2, 1)
    assert s.v.shape == (2, 1)


def test_lift_with_leaky_integrator():
    data = torch.ones(3, 2, 1)
    lifted = lift(li_step)
    z, s = lifted(
        data,
        state=LIState(
            v=torch.zeros(2, 1),
            i=torch.zeros(2, 1),
        ),
        input_weights=torch.ones(1, 1),
        p=LIFParameters(),
    )
    assert z.shape == (3, 2, 1)
    assert s.v.shape == (2, 1)
