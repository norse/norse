"""
Decoders as torch modules.
"""

import torch

from norse.torch.functional.decode import spike_time_decode


class SpikeTimeDecoder(torch.nn.Module):
    """
    Reduces an input tensor to the timestamp indices of spike occurences.
    Put differently, this decoder extracts the indices of spike activities
    so as to operate on time indices instead of spike patterns.

    Example:
    >>> x = torch.tensor([0, 0, 0, 1, 0])
    >>> spike_time_decode(x)
    >>> # tensor([[3]])

    >>> x = torch.tensor([[0, 1], [1, 1], [0, 0]])
    >>> spike_time_decode(x)
    >>> # tensor([[0, 1, 1],
    >>> #         [1, 0, 1]])

    >>> x = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=torch.float32, requires_grad=True)
    >>> spike_time_decode(x)
    >>> # torch.tensor([[0, 1, 1, 1, 2],
    >>> #               [1, 0, 1, 2, 2]])

    Arguments:
        tensor (torch.Tensor): A tensor of spikes (1's)

    Returns:
        A tensor of shape (``ndims``, ``nvalues``) where ``ndims`` is the number of
        dimensions in the tensor and ``nvalues`` are the spike-events that
        occured in the input tensor.
        Note that the returned tensor uses floats to support the flow of gradients.
    """

    @staticmethod
    def forward(input_tensor: torch.Tensor):
        return spike_time_decode(input_tensor)
