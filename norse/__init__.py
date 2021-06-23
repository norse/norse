"""Norse is a library for doing deep learning with spiking neural networks.
"""

from . import benchmark, dataset, task
from .torch import functional, models, module

import sys

try:
    import torch
    import norse_op

    setattr(sys.modules[__name__], "IS_OPS_LOADED", True)
    del torch  # Unload torch again to allow importing norse.torch
except ImportError or ModuleNotFoundError:
    setattr(sys.modules[__name__], "IS_OPS_LOADED", False)

__all__ = ["task", "benchmark", "dataset", "functional", "models", "module"]
