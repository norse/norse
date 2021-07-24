"""
A module for lifting neuron activation functions in time.
Simlar to the :module:`.lift`_ module.
"""

import torch


class _Lifted:
    """
    Helper class for the :func:`lift`_ function to allow for pickling.
    Used in distributed execution, like PyTorch Lightning
    """

    def __init__(self, activation, p=None):
        self.activation = activation
        self.p = p

    def __call__(self, input_tensor, **kwargs):
        if self.p is not None and "p" not in kwargs:
            kwargs["p"] = self.p

        state = kwargs.get("state")
        kwargs.pop("state", None)
        outputs = []
        for i in input_tensor:
            if state is not None:
                out, state = self.activation(i, state=state, **kwargs)
            else:
                out, state = self.activation(i, **kwargs)
            outputs.append(out)
        return torch.stack(outputs), state


def lift(activation, p=None):
    """
    Creates a lifted version of the given activation function which
    applies the activation function in the temporal domain. The returned
    callable can be applied later as if it was a regular activation function,
    but the input is now assumed to be a tensor whose first dimension is time.

    Parameters:
        activation (Callable[[torch.Tensor, Any, Any], Tuple[torch.Tensor, Any]]):
            The activation function that takes an input tensor, an optional state, and an
            optional parameter object and returns a tuple of (spiking output, neuron state).
            The returned spiking output includes the time domain.
        p (Any): An optional parameter object to hand to the activation function.

    Returns:
        A :class:`.Callable`_ that, when applied, evaluates the activation function N times,
        where N is the size of the outer (temporal) dimension. The application will provide
        a tensor of shape (time, ...).
    """

    return _Lifted(activation, p)
