"""
Internal utilities for modules
"""

import torch


def remove_autopses(weights):
    """
    Removes autopses by zeroing out the weight matrix diagonal.
    """
    return weights * (
        torch.ones_like(weights) - torch.eye(*weights.shape, device=weights.device)
    )
