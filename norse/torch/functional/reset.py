"""
Functions for reset mechanisms.
"""

from typing import Callable

import torch


def reset_value(
    z: torch.Tensor,
    v: torch.Tensor,
    reset: torch.Tensor,
    th: torch.Tensor,
) -> torch.Tensor:
    return v - z * (v + reset)


def reset_subtract(
    z: torch.Tensor, v: torch.Tensor, reset: torch.Tensor, th: torch.Tensor
) -> torch.Tensor:
    return v - z * th


def reset_by_method(
    z: torch.Tensor,
    v: torch.Tensor,
    reset: torch.Tensor,
    th: torch.Tensor,
    reset_method: str,
) -> torch.Tensor:
    if reset_method == "value":
        return reset_value(z, v, reset, th)
    if reset_method == "subtract":
        return reset_subtract(z, v, reset, th)
    raise ValueError(
        f"Unknown reset method: {reset_method}, must be 'value' or 'subtract'"
    )


def reset_fn_from_string(
    reset_method: str,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    if reset_method == "value":
        return reset_value
    if reset_method == "subtract":
        return reset_subtract
    raise ValueError(f"Unknown reset method: {reset_method}")
