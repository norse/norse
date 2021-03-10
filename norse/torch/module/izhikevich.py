from typing import Any, Optional, Tuple

import numpy as np
import torch

from norse.torch.functional.threshold import threshold

from ..functional.izhikevich import (
    IzhikevichState,
    IzhikevichParameters,
    izhikevich_step,
)

from from ..functional import izhikevich


class IzhikevichCell(torch.nn.Module):
    """Documentation WIP
    Cell for a Izhikevich model neuron population.

    Parameters:
        input_size (int): Size of the input. Also known as the number of input features.
        hidden_size (int): Size of the hidden state. Also known as the number of input features.
        p (IzhikevichParameters): Parameters of the Izhikevich neuron model.
        dt (float): Time step to use.
    Examples:
        >>> batch_size = 16
        >>> izhikevich = IzhikevichCell(10, 20)
        >>> input = torch.randn(batch_size, 10)
        >>> output, s0 = izhikevich(input)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: IzhikevichParameters = IzhikevichParameters(),
        spiking_method : str,
        dt: float = 0.001,
    ):
        super(IzhikevichCell, self).__init__()
        self.input_weights = torch.nn.Parameter(
            torch.randn(hidden_size, input_size) * np.sqrt(2 / hidden_size)
        )
        self.recurrent_weights = torch.nn.Parameter(
            torch.randn(hidden_size, hidden_size) * np.sqrt(2 / hidden_size)
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.p = p
        self.dt = dt
        
    def cell_initialisation(self, self.input_size, self.hidden_size, self.spiking_method : str):
        shape = (input_size, hidden_size)
        s, self.p = getattr(izhikevich, spiking_method)(shape) 

    def extra_repr(self):
        s = f"{self.input_size}, {self.hidden_size}, p={self.p}, dt={self.dt}"
        return s

    def forward(
        self, input_tensor: torch.Tensor, state: Optional[IzhikevichState] = None
    ) -> Tuple[torch.Tensor, IzhikevichState]:
        return izhikevich_step(
            input_current,
            s,
            p=self.p,
            dt=self.dt,
        )


class IzhikevichLayer(torch.nn.Module):
    """
    A neuron layer that wraps a recurrent IzhikevichCell in time such
    that the layer keeps track of temporal sequences of spikes.
    After application, the layer returns a tuple containing
      (spikes from all timesteps, state from the last timestep).
    Example:
        >>> data = torch.zeros(10, 5, 2) # 10 timesteps, 5 batches, 2 neurons
        >>> l = IzhikevichLayer(2, 4)
        >>> l(data) # Returns tuple of (Tensor(10, 5, 4), IzhikevichState)
    """

    def __init__(self, *cell_args, **kw_args):
        super(IzhikevichLayer, self).__init__()
        self.cell = IzhikevichCell(*cell_args, **kw_args)

    def forward(
        self, input_tensor: torch.Tensor, state: Optional[IzhikevichState] = None
    ) -> Tuple[torch.Tensor, IzhikevichState]:
        inputs = input_tensor.unbind(0)
        outputs = []  # torch.jit.annotate(List[torch.Tensor], [])
        for _, input_step in enumerate(inputs):
            out, state = self.cell(input_step, state)
            outputs += [out]
        # pytype: disable=bad-return-type
        return torch.stack(outputs), state
        # pytype: enable=bad-return-type
