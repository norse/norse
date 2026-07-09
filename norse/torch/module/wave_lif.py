"""
Klein-Gordon wave neuron module.

A neuron whose membrane potential follows second-order oscillatory dynamics
driven by the Klein-Gordon dispersion relation.

See :mod:`norse.torch.functional.wave_lif` for more information.
"""

import torch

from norse.torch.functional.wave_lif import (
    WaveLIFState,
    WaveLIFFeedForwardState,
    WaveLIFParameters,
    wave_lif_step,
    wave_lif_feed_forward_step,
)
from norse.torch.module.snn import SNN, SNNCell, SNNRecurrent, SNNRecurrentCell
from norse.torch.utils.clone import clone_tensor


class WaveLIFCell(SNNCell):
    r"""Module that computes a single leapfrog step of a Klein-Gordon wave
    neuron *without* recurrence and *without* time.

    .. math::
        v_{\text{new}} = \frac{2 v - (1 - \gamma\,dt/2)\,v_{\text{prev}}
        + dt^2 (-\chi^2 v + I_{\text{in}})}{1 + \gamma\,dt/2}

    Example:
        >>> data = torch.zeros(5, 2)  # 5 batches, 2 neurons
        >>> l = WaveLIFCell()
        >>> l(data)  # Returns tuple of (Tensor(5, 2), WaveLIFFeedForwardState)

    Arguments:
        p (WaveLIFParameters): Parameters of the wave neuron model.
        dt (float): Time step to use. Defaults to 0.001.
    """

    def __init__(self, p: WaveLIFParameters = WaveLIFParameters(), **kwargs):
        super().__init__(
            activation=wave_lif_feed_forward_step,
            state_fallback=self.initial_state,
            p=WaveLIFParameters(
                torch.as_tensor(p.chi),
                torch.as_tensor(p.gamma),
                torch.as_tensor(p.v_th),
                torch.as_tensor(p.v_reset),
                p.method,
                torch.as_tensor(p.alpha),
            ),
            **kwargs,
        )

    def initial_state(
        self, input_tensor: torch.Tensor
    ) -> WaveLIFFeedForwardState:
        state = WaveLIFFeedForwardState(
            v=clone_tensor(self.p.v_reset),
            v_prev=clone_tensor(self.p.v_reset),
        )
        state.v.requires_grad = True
        return state


class WaveLIFRecurrentCell(SNNRecurrentCell):
    r"""Module that computes a single leapfrog step of a Klein-Gordon wave
    neuron *with* recurrence and *without* time.

    Example:
        >>> data = torch.zeros(5, 2)  # 5 batches, 2 input features
        >>> l = WaveLIFRecurrentCell(2, 4)
        >>> l(data)  # Returns tuple of (Tensor(5, 4), WaveLIFState)

    Arguments:
        input_size (int): Size of the input.
        hidden_size (int): Size of the hidden state.
        p (WaveLIFParameters): Parameters of the wave neuron model.
        dt (float): Time step to use. Defaults to 0.001.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: WaveLIFParameters = WaveLIFParameters(),
        **kwargs,
    ):
        super().__init__(
            activation=wave_lif_step,
            state_fallback=self.initial_state,
            input_size=input_size,
            hidden_size=hidden_size,
            p=WaveLIFParameters(
                torch.as_tensor(p.chi),
                torch.as_tensor(p.gamma),
                torch.as_tensor(p.v_th),
                torch.as_tensor(p.v_reset),
                p.method,
                torch.as_tensor(p.alpha),
            ),
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> WaveLIFState:
        dims = (*input_tensor.shape[:-1], self.hidden_size)
        state = WaveLIFState(
            z=torch.zeros(
                dims,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            v=clone_tensor(self.p.v_reset),
            v_prev=clone_tensor(self.p.v_reset),
        )
        state.v.requires_grad = True
        return state


class WaveLIF(SNN):
    r"""Module that computes a sequence of leapfrog steps of a Klein-Gordon
    wave neuron *without* recurrence but *with* time.

    Example:
        >>> data = torch.zeros(10, 5, 2)  # 10 timesteps, 5 batches, 2 neurons
        >>> l = WaveLIF()
        >>> l(data)  # Returns tuple of (Tensor(10, 5, 2), WaveLIFFeedForwardState)

    Arguments:
        p (WaveLIFParameters): Parameters of the wave neuron model.
        dt (float): Time step to use. Defaults to 0.001.
    """

    def __init__(self, p: WaveLIFParameters = WaveLIFParameters(), **kwargs):
        super().__init__(
            activation=wave_lif_feed_forward_step,
            state_fallback=self.initial_state,
            p=WaveLIFParameters(
                torch.as_tensor(p.chi),
                torch.as_tensor(p.gamma),
                torch.as_tensor(p.v_th),
                torch.as_tensor(p.v_reset),
                p.method,
                torch.as_tensor(p.alpha),
            ),
            **kwargs,
        )

    def initial_state(
        self, input_tensor: torch.Tensor
    ) -> WaveLIFFeedForwardState:
        state = WaveLIFFeedForwardState(
            v=clone_tensor(self.p.v_reset),
            v_prev=clone_tensor(self.p.v_reset),
        )
        state.v.requires_grad = True
        return state


class WaveLIFRecurrent(SNNRecurrent):
    r"""Module that computes a sequence of leapfrog steps of a Klein-Gordon
    wave neuron *with* recurrence and *with* time.

    Example:
        >>> data = torch.zeros(10, 5, 2)  # 10 timesteps, 5 batches, 2 inputs
        >>> l = WaveLIFRecurrent(2, 4)
        >>> l(data)  # Returns tuple of (Tensor(10, 5, 4), WaveLIFState)

    Arguments:
        input_size (int): Size of the input.
        hidden_size (int): Size of the hidden state.
        p (WaveLIFParameters): Parameters of the wave neuron model.
        dt (float): Time step to use. Defaults to 0.001.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: WaveLIFParameters = WaveLIFParameters(),
        **kwargs,
    ):
        super().__init__(
            activation=wave_lif_step,
            state_fallback=self.initial_state,
            input_size=input_size,
            hidden_size=hidden_size,
            p=WaveLIFParameters(
                torch.as_tensor(p.chi),
                torch.as_tensor(p.gamma),
                torch.as_tensor(p.v_th),
                torch.as_tensor(p.v_reset),
                p.method,
                torch.as_tensor(p.alpha),
            ),
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> WaveLIFState:
        dims = (*input_tensor.shape[:-1], self.hidden_size)
        state = WaveLIFState(
            z=torch.zeros(
                dims,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            v=clone_tensor(self.p.v_reset),
            v_prev=clone_tensor(self.p.v_reset),
        )
        state.v.requires_grad = True
        return state
