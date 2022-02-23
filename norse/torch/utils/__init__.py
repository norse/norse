"""
Utilities for Norse networks in Torch.

Packages and subpackages may depend on Matplotlib and Tensorboard.
"""
import logging

try:
    from .plot import *
except ImportError as e:
    logging.debug(f"Failed to import Norse plotting utilities: {e}")

try:
    from .tensorboard import *
except ImportError as e:
    logging.debug(f"Failed to import Norse plotting utilities: {e}")

del logging
