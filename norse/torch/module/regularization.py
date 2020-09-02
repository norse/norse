"""
This module contains ``torch.nn.Module``s for regularisation operations on
spiking layers, where it can be desirable to regularise spikes,
membrane parameters, or other properties over time.
"""
import torch

from norse.torch.functional.regularization import (
    regularize_step,
    spike_accumulator,
    T,
    Accumulator,
)


class RegularizationCell(torch.nn.Module):
    """
    A regularisation cell that accumulates some state (for instance number of spikes)
    for each forward step, which can later be applied to a loss term.

    Example:
        >>> import torch
        >>> from norse.torch.module import lif, regularization
        >>> cell = lif.LIFCell(2, 4) # 2 -> 4
        >>> r = regularization.RegularizationCell() # Defaults to spike counting
        >>> data = torch.ones(5, 2)  # Batch size of 5
        >>> z, s = cell(data)
        >>> z, regularization_term = r(z, s)
        >>> ...
        >>> loss = ... + 1e-3 * regularization_term

    Parameters:
        accumulator (Accumulator):
            The accumulator that aggregates some data (such as spikes) that can later
            be included in an error term.
        state (Optional[T]): The regularization state to be aggregated to of any type T. Defaults to None.
    """

    def __init__(self, accumulator: Accumulator = spike_accumulator, state: T = None):
        super(RegularizationCell, self).__init__()
        self.accumulator = accumulator
        self.state = state

    def forward(self, z, s):
        _, state = regularize_step(z, s, self.accumulator, self.state)
        self.state = state
        return z, self.state
