"""
Utilities for Norse networks in Torch.

Packages and subpackages may depend on Matplotlib and Tensorboard.
"""

from .tensorboard import (
    hook_spike_activity_mean,
    hook_spike_activity_sum,
    hook_spike_histogram_mean,
    hook_spike_histogram_sum,
    hook_spike_image,
)

__all__ = [
    "hook_spike_activity_mean",
    "hook_spike_activity_sum",
    "hook_spike_histogram_mean",
    "hook_spike_histogram_sum",
    "hook_spike_image",
]
