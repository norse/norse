import torch


def heaviside(input):
    return 0.5 * (torch.sign(input) + 1.0)
