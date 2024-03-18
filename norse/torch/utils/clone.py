from typing import Union, Optional
from numbers import Number
import torch


def clone_tensor(x: Union[torch.Tensor, Number], device: Optional[str] = None):
    cloned = torch.as_tensor(x)
    if isinstance(x, torch.Tensor):
        cloned = x.detach().clone()
    else:
        raise ValueError("Expected tensor or number, but received ", x)
    if device is not None:
        return cloned.to(device)
    else:
        return cloned
