"""Norse is a library for doing deep learning with spiking neural networks.
"""

from . import task

from .torch import benchmark, functional, models, module
from .torch.models import conv
from .torch.module import lif, lsnn

__all__ = [
    task,
    benchmark,
    functional,
    conv,
    module.lif,
    module.lsnn,
]
