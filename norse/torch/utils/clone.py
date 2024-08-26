from typing import Union, Optional
from numbers import Number
import torch


def clone_tensor(x: torch.Tensor, device: Optional[str] = None) -> torch.Tensor:
    """Clone a tensor and move it to a device, if specified."""
    cloned = torch.as_tensor(x).detach().clone()
    if device is not None:
        return cloned.to(device)
    else:
        return cloned
