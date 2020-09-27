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
from typing import Any, Callable, NewType, Optional, TypeVar
import torch

# The type of regularisation state, e. g. ints for accumulating spikes or
# torch.Tensor for accumulating membrane potential.
T = TypeVar("T")

# An accumulator that can take some cell output, cell state, and regularization state
# and return an updated regularization state
Accumulator = NewType("Accumulator", Callable[[torch.Tensor, Any, Optional[T]], T])


def spike_accumulator(z: torch.Tensor, s: Any, state: int = 0) -> int:
    """
    A spike accumulator that aggregates spikes and returns the total sum as an integer.

    Parameters:
        z (torch.Tensor): Spikes from some cell
        s (Any): Cell state
        state (Optional[int]): The regularization state to be aggregated to. Defaults to 0.

    Returns:
        A new RegularizationState containing the aggregated data
    """
    return state + z.sum()


def voltage_accumulator(
    z: torch.Tensor, s: Any, state: Optional[torch.Tensor] = None
) -> int:
    """
    A spike accumulator that aggregates membrane potentials over time. Requires that the
    input state ``s`` has a ``v`` property.

    Parameters:
        z (torch.Tensor): Spikes from some cell
        s (Any): Cell state
        state (Optional[torch.Tensor]): The regularization state to be aggregated to.
        Defaults to zeros for all entries.

    Returns:
        A new RegularizationState containing the aggregated data
    """
    if not state:
        state = torch.zeros_like(s.v)
    return state + s.v


def regularize_step(
    z: torch.Tensor,
    s: Any,
    accumulator: Accumulator = spike_accumulator,
    state: T = None,
) -> (torch.Tensor, T):
    """
    Takes one step for a regularizer that aggregates some information (based on the
    spike_accumulator function), which is pushed forward and returned for future
    inclusion in an error term.

    Parameters:
        z (torch.Tensor): Spikes from a cell
        s (Any): Neuron state
        state (Optional[T]): The regularization state to be aggregated to of any type T. Defaults to None

    Return:
        A tuple of (spikes, regularizer state)
    """
    if not state:
        return z, accumulator(z, s)
    return z, accumulator(z, s, state)
