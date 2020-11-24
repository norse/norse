import torch

from norse.torch.functional.lif import LIFFeedForwardState, lif_feed_forward_step
from norse.torch.functional.regularization import regularize_step, voltage_accumulator


def test_regularisation_spikes():
    x = torch.ones(5, 10)
    s = LIFFeedForwardState(torch.ones(10), torch.ones(10))
    z, s = lif_feed_forward_step(x, s)
    zr, rs = regularize_step(z, s)
    assert torch.equal(z, zr)
    assert rs == 0
    z, s = lif_feed_forward_step(x, s)
    zr, rs = regularize_step(z, s)
    assert rs == 50


def test_regularisation_voltage():
    x = torch.ones(5, 10)
    s = LIFFeedForwardState(torch.ones(10), torch.ones(10))
    z, s = lif_feed_forward_step(x, s)
    # pytype: disable=wrong-arg-types
    zr, rs = regularize_step(z, s, accumulator=voltage_accumulator)
    # pytype: enable=wrong-arg-types
    assert torch.equal(z, zr)
    assert torch.equal(s.v, rs)


def test_regularisation_voltage_state():
    x = torch.ones(5, 10)
    state = torch.zeros(10)
    s = LIFFeedForwardState(torch.ones(10), torch.ones(10))
    z, s = lif_feed_forward_step(x, s)
    # pytype: disable=wrong-arg-types
    zr, rs = regularize_step(z, s, accumulator=voltage_accumulator, state=state)
    # pytype: enable=wrong-arg-types
    assert torch.equal(z, zr)
    assert torch.equal(s.v, rs)
