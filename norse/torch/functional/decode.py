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
