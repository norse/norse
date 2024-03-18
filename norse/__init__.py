"""Norse is a library for doing deep learning with spiking neural networks.
"""

from importlib.metadata import version, PackageNotFoundError
from norse import benchmark, dataset, task, torch, utils

__all__ = ["benchmark", "dataset", "task", "torch", "utils"]


try:
    __version__ = version("norse")
except PackageNotFoundError:
    # package is not installed
    pass
