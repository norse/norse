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


class SpikeTimeDecode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i: torch.Tensor):
        if i.is_sparse:
            sparse = i.coalesce()
        else:
            sparse = i.to_sparse().coalesce()
        ctx.save_for_backward(sparse, torch.as_tensor(i.is_sparse))
        indices = sparse.indices().float()
        if i.requires_grad:
            indices.requires_grad = True
        return indices

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        result, was_sparse = ctx.saved_tensors
        values = result.values() * grad_output.mean(0)  # Mean over index dimension
        sparse = torch.sparse_coo_tensor(
            indices=result.indices(), values=values, size=result.size()
        )
        if was_sparse:
            return sparse
        else:
            return sparse.to_dense()


def spike_time_decode(tensor: torch.Tensor):
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
    return SpikeTimeDecode.apply(tensor)
