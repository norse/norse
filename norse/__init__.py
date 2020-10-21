"""Norse is a library for doing deep learning with spiking neural networks.
"""

from . import benchmark, dataset, task

# Attempt import of optimized code
try:
    import torch
    from pathlib import Path

    torch.ops.load_library(Path(__file__).resolve().parent.parent / "norse_op.so")
    import sys

    setattr(sys.modules[__name__], "IS_OPS_LOADED", True)
except OSError:
    import sys

    setattr(sys.modules[__name__], "IS_OPS_LOADED", False)

from .torch import functional, models, module

__all__ = ["task", "benchmark", "dataset", "functional", "models", "module"]
