"""Norse is a library for doing deep learning with spiking neural networks.
"""

from . import benchmark, dataset, task
from .torch import functional, models, module, util

__all__ = ["task", "benchmark", "dataset", "functional", "models", "module", "util"]
