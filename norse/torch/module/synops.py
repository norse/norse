"""
Module wrapper for measuring synaptic operations (SynOps) during the forward
pass of a spiking neural network.
"""

from typing import Optional, Tuple, Union

import torch
from torch import nn

from norse.torch.functional.synops import synops_accumulator


class SynOpsCounter(nn.Module):
    """
    Counts synaptic operations (SynOps) produced by spikes passing through a
    weight layer.

    Wraps a single synaptic layer (``nn.Linear`` or ``nn.Conv2d``) and
    accumulates the total number of multiply-accumulate events triggered by
    spikes across forward steps.

    Example:
        >>> import torch
        >>> from torch import nn
        >>> import norse.torch as snn
        >>> cell = snn.LIFCell()
        >>> counter = snn.SynOpsCounter(nn.Linear(2, 8))
        >>> data = torch.ones(5, 2) + 10   # drives all neurons to fire
        >>> z, s = cell(data)
        >>> z, total_synops = counter(z)
        >>> print(total_synops)            # synops accumulated so far

    Parameters:
        layer (nn.Module): The synaptic layer whose operations are counted.
            Must be ``nn.Linear`` or ``nn.Conv2d``.
        state (Optional): Initial accumulated synops value. Defaults to ``None``
            (treated as 0 on first call).
    """

    def __init__(
        self,
        layer: nn.Module,
        state: Optional[Union[int, torch.Tensor]] = None,
    ):
        super().__init__()
        self.layer = layer
        self.state = state

    def reset(self) -> None:
        """Resets the accumulated synops counter to zero."""
        self.state = None

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Union[int, torch.Tensor]]:
        """
        Accumulates synops for the current timestep's spikes.

        Parameters:
            z (torch.Tensor): Spike tensor from the preceding spiking layer.

        Returns:
            A tuple of ``(z, accumulated_synops)`` where ``z`` is unchanged
            and ``accumulated_synops`` is the running total including this step.
        """
        self.state = synops_accumulator(z, self.layer, self.state)
        return z, self.state
