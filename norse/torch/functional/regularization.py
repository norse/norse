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

    Raises:
        ValueError: If ``state`` already has the same number of dimensions as
            ``s.v`` (i.e. it's a previously-accumulated state, not a reduced-
            rank broadcasting initializer) but a different concrete shape --
            e.g. accumulated against a different batch size. ``state`` is
            cached from whichever call first created it (always via
            ``torch.zeros_like(s.v)``, same rank as ``s.v``), so a later call
            with a different same-rank shape would otherwise either crash
            with an opaque broadcasting error, or -- if the shapes happen to
            be broadcastable (e.g. batch size 1) -- silently combine
            unrelated samples into the accumulated state with no error at
            all. A genuinely reduced-rank initial ``state`` (e.g. a per-
            neuron-only ``state`` meant to broadcast up on first use, as in
            ``test_regularisation_voltage_state``) is NOT an error -- that's
            a legitimate initialization pattern, not an accumulated mismatch.
            Call with ``state=None`` (or :meth:`RegularizationCell.reset`) to
            start a new accumulation.
    """
    if state is None:
        state = torch.zeros_like(s.v)
    elif state.dim() == s.v.dim() and state.shape != s.v.shape:
        raise ValueError(
            f"voltage_accumulator: state has shape {tuple(state.shape)} but this "
            f"call's s.v has shape {tuple(s.v.shape)} -- state was accumulated "
            f"against a different shape (e.g. a different batch size) and cannot "
            f"be combined with this call's values. Pass state=None, or call "
            f"RegularizationCell.reset(), to start a new accumulation."
        )
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
