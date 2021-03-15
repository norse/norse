# from typing import Optional, Tuple

import torch

from ..functional.izhikevich import (
    IzhikevichState,
    IzhikevichSpikingBehaviour,
    izhikevich_step,
)

from norse.torch.module.snn import SNNCell  # , SNN, SNNRecurrent, SNNRecurrentCell


class IzhikevichCell(SNNCell):
    """Documentation WIP
    Module that computes a single Izhikevich neuron-model *without* recurrence and *without* time.

    Parameters:
        spiking_method (IzhikevichSpikingBehaviour) : parameters and initial state of the neuron
    Example with tonic spiking:
        >>> from norse.torch import IzhikevichCell, tonic_spiking
        >>> batch_size = 16
        >>> cell = IzhikevichCell(tonic_spiking)
        >>> input = torch.randn(batch_size, 10)
        >>> output, s0 = cell(input)
    """

    def __init__(self, spiking_method: IzhikevichSpikingBehaviour, **kwargs):
        super().__init__(
            izhikevich_step, self.initial_state, spiking_method.p, **kwargs
        )
        self.spiking_method = spiking_method

    def initial_state(self, input_tensor: torch.Tensor) -> IzhikevichState:
        state = self.spiking_method.s
        return state
