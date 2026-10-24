import pytest
import torch

from norse.torch.functional.lif import (
    LIFFeedForwardState,
    LIFParameters,
    lif_feed_forward_step,
)
from norse.torch.functional.regularization import regularize_step, voltage_accumulator


def test_regularisation_spikes():
    x = torch.ones(5, 10)
    s = LIFFeedForwardState(torch.ones(10), torch.ones(10))
    z, s = lif_feed_forward_step(x, s, LIFParameters())
    zr, rs = regularize_step(z, s)
    assert torch.equal(z, zr)
    assert rs == 50
    z, s = lif_feed_forward_step(x, s, LIFParameters())
    zr, rs = regularize_step(z, s)
    assert rs == 0


def test_regularisation_voltage():
    x = torch.ones(5, 10)
    s = LIFFeedForwardState(torch.ones(10), torch.ones(10))
    z, s = lif_feed_forward_step(x, s, LIFParameters())
    # pytype: disable=wrong-arg-types
    zr, rs = regularize_step(z, s, accumulator=voltage_accumulator)
    # pytype: enable=wrong-arg-types
    assert torch.equal(z, zr)
    assert torch.equal(s.v, rs)


def test_regularisation_voltage_state():
    x = torch.ones(5, 10)
    state = torch.zeros(10)
    s = LIFFeedForwardState(torch.ones(10), torch.ones(10))
    z, s = lif_feed_forward_step(x, s, LIFParameters())
    # pytype: disable=wrong-arg-types
    zr, rs = regularize_step(z, s, accumulator=voltage_accumulator, state=state)
    # pytype: enable=wrong-arg-types
    assert torch.equal(z, zr)
    assert torch.equal(s.v, rs)


def test_regularisation_voltage_rejects_batch_size_mismatch():
    """An accumulated state (same rank as s.v, i.e. already batch-shaped
    from a previous call) must not silently combine with a different
    batch size's s.v -- not a crash with an opaque broadcasting error, and
    NOT a silent wrong value when the sizes happen to be broadcastable."""
    x5 = torch.ones(5, 10)
    s = LIFFeedForwardState(torch.ones(10), torch.ones(10))
    _, s = lif_feed_forward_step(x5, s, LIFParameters())
    # state accumulated against batch size 5 (same rank as s.v: dim()==2)
    accumulated_state = torch.zeros_like(s.v)

    x1 = torch.ones(1, 10)
    s1 = LIFFeedForwardState(torch.ones(10), torch.ones(10))
    _, s1 = lif_feed_forward_step(x1, s1, LIFParameters())
    # batch size 1 broadcasts against batch size 5 with no shape error --
    # this must still be rejected, not silently combined
    with pytest.raises(ValueError):
        regularize_step(
            torch.ones(1), s1, accumulator=voltage_accumulator, state=accumulated_state
        )


def test_regularisation_voltage_allows_reduced_rank_initial_state():
    """A reduced-rank initial state (fewer dims than s.v, e.g. a per-neuron-
    only zero vector meant to broadcast up on first use) is a legitimate
    initialization pattern, NOT the accumulated-batch-mismatch bug -- must
    stay allowed (this is exactly test_regularisation_voltage_state's
    existing pattern, asserted here explicitly as a contract)."""
    x = torch.ones(5, 10)
    state = torch.zeros(10)  # dim()==1, s.v will be dim()==2
    s = LIFFeedForwardState(torch.ones(10), torch.ones(10))
    z, s = lif_feed_forward_step(x, s, LIFParameters())
    zr, rs = regularize_step(z, s, accumulator=voltage_accumulator, state=state)
    assert torch.equal(s.v, rs)
