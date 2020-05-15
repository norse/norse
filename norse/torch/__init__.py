"""Norse is a library for doing deep learning with spiking neural networks.

This package contains modules that extends PyTorch with spiking neural
network functionality.
"""

from . import functional
from . import module

__all__ = [functional, module]
