import torch
from .heaviside import heaviside


def heaviside_test():
    assert heaviside(torch.ones(100)) == torch.ones(100)
    assert heaviside(-1.0 * torch.ones(100)) == torch.zeros(100)
