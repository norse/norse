from typing import Tuple, Union


import torch
from norse.torch.utils.state import _is_module_stateful


class Lift(torch.nn.Module):
    """Lift applies a given torch.nn.Module over
       a temporal sequence. In other words this module
       applies the given torch.nn.Module N times, where N
       is the outer dimension in the provided tensor.

    Parameters:
        module: Module to apply

    Examples:

        >>> batch_size = 16
        >>> seq_length = 1000
        >>> in_channels = 64
        >>> out_channels = 32
        >>> conv2d = Lift(torch.nn.Conv2d(in_channels, out_channels, 5, 1))
        >>> data = torch.randn(seq_length, batch_size, 20, 30)
        >>> output = conv2d(data)


        >>> data = torch.randn(seq_length, batch_size, in_channels, 20, 30)
        >>> module = torch.nn.Sequential(
        >>>     Lift(torch.nn.Conv2d(in_channels, out_channels, 5, 1)),
        >>>     LIF(),
        >>> )
        >>> output, _ = module(data)
    """

    def __init__(self, module: torch.nn.Module):
        super(Lift, self).__init__()
        self.lifted_module = module
        self.is_stateful = _is_module_stateful(module)
        # or isinstance(
        #    module, torch.nn.SequentialState
        # )

    def forward(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """Apply the module over the input along the 0-th (time) dimension
        and accumulate the outputs in an output tensor.

        Parameters:
            x : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

        Note:
            If the input is a tuple of two tensors, the second tuple entry will be ignored.
        """
        state = None
        if isinstance(x, tuple):
            x, state = x

        T = x.shape[0]
        outputs = []

        for ts in range(T):
            if self.is_stateful:
                out, state = self.lifted_module(x[ts], state)
            else:
                out = self.lifted_module(x[ts])
            outputs += [out]

        if self.is_stateful:
            return torch.stack(outputs), state
        else:
            return torch.stack(outputs)
