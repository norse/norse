import torch
from norse.torch.functional.heaviside import heaviside


def test_heaviside():
    assert torch.equal(heaviside(torch.ones(100)), torch.ones(100))
    assert torch.equal(heaviside(-1.0 * torch.ones(100)), torch.zeros(100))
