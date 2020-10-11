import torch

from norse.torch.functional.logical import (
    logical_and,
    logical_or,
    logical_xor,
    muller_c,
)
from norse.torch.functional.logical import posedge_detector


def test_logical_and():
    z = logical_and(torch.as_tensor([1, 0, 0, 1]), torch.as_tensor([1, 0, 0, 0]))
    assert torch.equal(z, torch.as_tensor([1, 0, 0, 0]))


def test_logical_or():
    z = logical_or(torch.as_tensor([1, 0, 0, 1]), torch.as_tensor([1, 0, 0, 0]))
    assert torch.equal(z, torch.as_tensor([1, 0, 0, 1]))


def test_logical_xor():
    z = logical_xor(torch.as_tensor([1, 0, 0, 1]), torch.as_tensor([1, 0, 0, 0]))
    assert torch.equal(z, torch.as_tensor([0, 0, 0, 1]))


def test_muller_c():
    z = muller_c(
        torch.as_tensor([1, 0, 0, 1]),
        torch.as_tensor([1, 0, 0, 0]),
        torch.as_tensor([1, 0, 0, 0]),
    )
    assert torch.equal(z, torch.as_tensor([1, 0, 0, 0]))


def test_posedge_detector():
    z = torch.as_tensor([1, 0, 0, 1])
    z_prev = torch.as_tensor([0, 0, 1, 1])
    z_out = posedge_detector(z, z_prev)
    assert torch.equal(z_out, torch.as_tensor([1, 0, 0, 0]))
