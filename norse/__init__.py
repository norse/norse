"""Norse is a library for doing deep learning with spiking neural networks.
"""

from . import benchmark, dataset, task

import torch
from pathlib import Path

torch.ops.load_library(Path(__file__).resolve().parent.parent / 'norse_op.so')

from .torch import functional
from .torch.models import conv
from .torch.module import lif, lsnn

__all__ = [
    "task",
    "benchmark",
    "dataset",
    "functional",
    "conv",
    "lif",
    "lsnn",
]
