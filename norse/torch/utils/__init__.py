"""
Utilities for Norse networks in Torch.

Packages and subpackages may depend on Matplotlib and Tensorboard.
"""
import logging
import torch
from typing import Union, Optional
from numbers import Number

from .import_nir import from_nir
from .export_nir import to_nir
from .clone import clone_tensor

try:
    from .plot import *
except ImportError as e:
    logging.debug(f"Failed to import Norse plotting utilities: {e}")

try:
    from .tensorboard import *
except ImportError as e:
    logging.debug(f"Failed to import Norse plotting utilities: {e}")

del logging
