"""
Utilities for Norse networks in Torch.

Packages and subpackages may depend on Matplotlib and Tensorboard.
"""
import logging
import torch
from typing import Union, Optional
from numbers import Number

try:
    from .plot import *
except ImportError as e:
    logging.debug(f"Failed to import Norse plotting utilities: {e}")

try:
    from .tensorboard import *
except ImportError as e:
    logging.debug(f"Failed to import Norse plotting utilities: {e}")

del logging


def clone_tensor(x: Union[torch.Tensor, Number], device: Optional[str] = None):
    if isinstance(x, torch.Tensor):
        cloned = x.detach().clone()
    elif isinstance(x, Number):
        cloned = torch.as_tensor(x)
    else:
        raise ValueError("Expected tensor or number, but received ", x)
    if device is not None:
        return cloned.to(device)
    else:
        return cloned
