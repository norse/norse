"""
This module contains functional components for regularization operations on
spiking layers, where it can be desirable to regularize spikes,
membrane parameters, or other properties over time.

In this functional version, the aim is to collect some state ``s`` for each
forward step. The collection depends on the output of the layer which, by
default, simply just counts spikes. It is the job of the user to include the
regularization in an error term later.

Read more on `Wikipedia <https://en.wikipedia.org/wiki/Regularization_%28mathematics%29>`_.
"""
from typing import Any, TypedDict, Callable
import torch


class RegularizationState(TypedDict):
    """
    Stateful representation of a single regularisation step.
    Equivalent to a typed dictionary where any information can be stored in principle.
    """


def spike_accumulator(
    z: torch.Tensor, s: Any, state: RegularizationState
) -> RegularizationState:
    """
    A spike accumulator that aggregates spikes and stores the total number under `num_spikes`.

    Parameters:
        z (torch.Tensor): Spikes from some cell
        s (Any): Cell state
        state (RegularizationState): The regularization state to be aggregated to

    Returns:
        A new RegularizationState containing the aggregated data
    """
    return RegularizationState(num_spikes=state['num_spikes'] + z.sum())


def regularize_step(
    z: torch.Tensor,
    s: Any,
    accumulator: Callable[
        [torch.Tensor, Any, RegularizationState], RegularizationState
    ] = spike_accumulator,
    state=RegularizationState(num_spikes=0),
):
    """
    Takes one step for a regularizer that aggregates some information (based on the
    spike_accumulator function), which is pushed forward and returned for future
    inclusion in an error term.

    Parameters:
        z (torch.Tensor): Spikes from a cell
        s (Any): Neuron state
        state (RegularizationState): The regularization state to be aggregated to

    Return:
        A tuple of (spikes, state, RegularizationState)
    """
    return z, s, accumulator(z, s, state)
