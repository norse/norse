from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch

from norse.torch.functional.threshold import threshold

from ..functional.izhikevich import (
    IzhikevichState,
    IzhikevichParameters,
    IzhikevichSpikingBehaviour,
    izhikevich_step,
    pass_state,
)

from ..functional import izhikevich

from norse.torch.module.snn import SNN, SNNCell, SNNRecurrent, SNNRecurrentCell


class IzhikevichCell(SNNCell):
    """Documentation WIP
    Module that computes a single Izhikevich neuron-model *without* recurrence and *without* time.

    Parameters:
        spiking_method (IzhikevichSpikingBehaviour) : parameters and initial state of the neuron
    Examples:
        >>> batch_size = 16
        >>> izhikevich = IzhikevichCell(10, 20)
        >>> input = torch.randn(batch_size, 10)
        >>> output, s0 = izhikevich(input)
    """

    def __init__(self, spiking_method: IzhikevichSpikingBehaviour, **kwargs):
        super().__init__(
            izhikevich_step, self.initial_state, spiking_method.p, **kwargs
        )
        self.spiking_method = spiking_method

    def initial_state(self, input_tensor: torch.Tensor) -> IzhikevichState:
        state = self.spiking_method.s
        return state


class Izhikevich(SNN):
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
