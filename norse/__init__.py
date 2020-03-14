"""Norse is a library for doing deep learning with spiking neural networks.
"""

from . import torch
from . import task

__all__ = [
    task,
    torch.benchmark,
    torch.functional,
    torch.models.conv,
    torch.module.lif,
    torch.module.lsnn,
]
