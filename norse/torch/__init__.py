"""Norse is a library for doing deep learning with spiking neural networks.

This package contains modules that extends PyTorch with spiking neural
network functionality.
"""
import logging

from .models import *
from .module import *

try:
    from .utils import *
except ImportError as e:
    logging.debug("Failed to import Norse utilities", e)

del logging
