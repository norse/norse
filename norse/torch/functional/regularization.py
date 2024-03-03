"""
This module contains functional components for regularization operations on
spiking layers, where it can be desirable to regularize spikes,
membrane parameters, or other properties over time.

In this functional module, the aim is to collect some state ``s`` for each
forward step. The collection depends on the output of the layer which, by
default, simply just counts spikes. It is the job of the user to include the
regularization in an error term later.

Read more on `Wikipedia <https://en.wikipedia.org/wiki/Regularization_%28mathematics%29>`_.
"""

from typing import Any, Callable, NewType, Optional, Tuple

import torch

# An accumulator that can take some cell output, cell state, and regularization state
# and return an updated regularization state
Accumulator = NewType("Accumulator", Callable[[torch.Tensor, Any, Optional[Any]], Any])


def spike_accumulator(
    z: torch.Tensor, _: Any, state: Optional[torch.Tensor] = None
) -> int:
    """
    A spike accumulator that aggregates spikes and returns the total sum as an integer.

    Parameters:
        z (torch.Tensor): Spikes from some cell
        s (Any): Cell state
        state (Optional[int]): The regularization state to be aggregated to. Defaults to 0.

    Returns:
        A new RegularizationState containing the aggregated data
    """
    if state is None:
        state = 0
    return state + z.sum()


def voltage_accumulator(
    z: torch.Tensor, s: Any, state: Optional[torch.Tensor] = None
) -> int:
    """
    A spike accumulator that aggregates membrane potentials over time. Requires that the
    input state ``s`` has a ``v`` property (for voltage).

    Parameters:
        z (torch.Tensor): Spikes from some cell
        s (Any): Cell state
        state (Optional[torch.Tensor]): The regularization state to be aggregated to.
        Defaults to zeros for all entries.

    Returns:
        A new RegularizationState containing the aggregated data
    """
    if state is None:
        state = torch.zeros_like(s.v)
    return state + s.v


# pytype: disable=annotation-type-mismatch
def regularize_step(
    z: torch.Tensor,
    s: Any,
    accumulator: Accumulator = spike_accumulator,
    state: Optional[Any] = None,
) -> Tuple[torch.Tensor, Any]:
    """
    Takes one step for a regularizer that aggregates some information (based on the
    spike_accumulator function), which is pushed forward and returned for future
    inclusion in an error term.

    Parameters:
        z (torch.Tensor): Spikes from a cell
        s (Any): Neuron state
        accumulator (Accumulator): Accumulator that decides what should be accumulated
        state (Optional[Any]): The regularization state to be aggregated. Typically some numerical value like a count. Defaults to None

    Return:
        A tuple of (spikes, regularizer state)
    """
    return z, accumulator(z, s, state)


# pytype: enable=annotation-type-mismatch
