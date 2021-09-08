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
    >>> SpikeTimeDecoder()(x)
    tensor([[3]])

    >>> import norse.torch as nt
    >>> model = nt.SequentialState(
    >>>     nt.LIFCell(),
    >>>     nt.SpikeTimeDecoder()
    >>> )
    >>> x, _ = model(torch.ones(10, 2))
    tensor([[6, 6, 9, 9]
            [0, 1, 0, 1]])
    """

    @staticmethod
    def forward(input_tensor: torch.Tensor):
        return spike_time_decode(input_tensor)
