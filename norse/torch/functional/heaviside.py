import torch


def heaviside(input):
    return torch.where(
        input <= torch.zeros_like(input),
        torch.zeros_like(input),
        torch.ones_like(input),
    )
    # return 0.5 * (torch.sign(input) + 1.0)
