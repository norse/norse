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


class FirstSpikeTimeDecode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor) -> torch.Tensor:
        first_spike = torch.argmax(input_tensor, dim=0)
        ctx.shape = input_tensor.shape
        ctx.save_for_backward(first_spike)
        return first_spike

    @staticmethod
    def backward(ctx, grad_output):
        grad_in = torch.zeros_like(ctx.shape)
        (first_spike,) = ctx.saved_tensors
        grad_in[first_spike] = grad_output
        return grad_in


def first_spike_time(spikes: torch.Tensor):
    return FirstSpikeTimeDecode.apply(spikes)
