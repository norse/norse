"""
Stateless decoding functionality for Norse, where events in time are
converted to numerical representations. Many of the decoders are simply
mirrors of the
`PyTorch Tensor aggregation functions <https://pytorch.org/docs/stable/tensors.html>`_,
with the difference that they typically aggregate the first (temporal) dimension.
They are therefore useful to inject in :meth:`norse.torch.functional.lift` ed
modules or situations where you need pickled functions.
"""
import torch


def sum_decode(tensor: torch.Tensor, dimension: int = 0):
    """
    Sums the input tensor in the first dimension by default.
    """
    return tensor.sum(dimension)


def spike_time_decode(tensor: torch.Tensor):
    """
    Reduces an input tensor to the timestamp indices of spike occurences.
    Put differently, this decoder extracts the indices of spike activities
    so as to operate on time indices instead of spike patterns.

    Example:
    >>> x = torch.tensor([0, 0, 0, 1, 0])
    >>> spike_time_decode(x)
    tensor([[3]])

    >>> x = torch.tensor([[0, 1], [1, 1], [0, 0]])
    >>> spike_time_decode(x)
    tensor([[0, 1, 1],
            [1, 0, 1]])

    Arguments:
        tensor (torch.Tensor): A tensor of spikes (1's)

    Returns:
        A tensor of shape (``ndims``, ``nvalues``) where ``ndims`` is the number of
        dimensions in the tensor and ``nvalues`` are the number of spike-events that
        occured in the input tensor.
    """
    if tensor.is_sparse:
        sparse = tensor
    else:
        sparse = tensor.to_sparse()
    return sparse.coalesce().indices()
