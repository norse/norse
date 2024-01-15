"""
Functions for reset mechanisms.
"""

from typing import Callable
import torch

ResetMethod = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]


def reset_value(
    z: torch.Tensor,
    v: torch.Tensor,
    reset: torch.Tensor,
    th: torch.Tensor,
) -> torch.Tensor:
    return (1 - z) * v + z * reset


def reset_subtract(
    z: torch.Tensor, v: torch.Tensor, reset: torch.Tensor, th: torch.Tensor
) -> torch.Tensor:
    return (1 - z) * v + z * (v - th)
