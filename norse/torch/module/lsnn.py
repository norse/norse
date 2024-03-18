"""
Long-short term memory module, building on the work by
[G. Bellec, D. Salaj, A. Subramoney, R. Legenstein, and W. Maass](https://github.com/IGITUGraz/LSNN-official).

See :mod:`norse.torch.functional.lsnn` for more information.
"""

import torch

from norse.torch.functional.lsnn import (
    LSNNParameters,
    LSNNState,
    LSNNFeedForwardState,
    lsnn_step,
    lsnn_feed_forward_step,
)
from norse.torch.functional.adjoint.lsnn_adjoint import (
    lsnn_adjoint_step,
    lsnn_feed_forward_adjoint_step,
)
from norse.torch.module.snn import SNNCell, SNNRecurrentCell, SNN, SNNRecurrent


class LSNNCell(SNNCell):
    r"""Euler integration cell for LIF Neuron with threshold adaptation
    *without* recurrence.
    More specifically it implements one integration step of the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i \\
            \dot{b} &= -1/\tau_{b} b
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}} + b)

    and transition equations

    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\text{reset}} \\
            i &= i + \text{input} \\
            b &= b + \beta z
        \end{align*}

    Parameters:
        p (LSNNParameters): parameters of the lsnn unit
        p (torch.nn.Module): The neuron parameters as a torch Module, which allows the module
            to configure neuron parameters as optimizable.
        dt (float): Time step to use in integration. Defaults to 0.001.
    """

    def __init__(
        self, p: LSNNParameters = LSNNParameters(), adjoint: bool = False, **kwargs
    ):
        if adjoint:
            super().__init__(
                activation=lsnn_feed_forward_adjoint_step,
                state_fallback=self.initial_state,
                p=p,
                **kwargs,
            )
        else:
            super().__init__(
                activation=lsnn_feed_forward_step,
                state_fallback=self.initial_state,
                p=p,
                **kwargs,
            )

    def initial_state(self, input_tensor: torch.Tensor) -> LSNNFeedForwardState:
        state = LSNNFeedForwardState(
            v=torch.full(
                input_tensor.shape,
                self.p.v_leak.detach(),
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            i=torch.zeros(
                *input_tensor.shape,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            b=self.p.v_th.detach(),
        )
        state.v.requires_grad = True
        return state


class LSNNRecurrentCell(SNNRecurrentCell):
    r"""Module that computes a single euler-integration step of a LSNN
    neuron-model *with* recurrence. More specifically it implements one
    integration step of the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i \\
            \dot{b} &= -1/\tau_{b} b
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}} + b)

    and transition equations

    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\text{reset}} \\
            i &= i + w_{\text{input}} z_{\text{in}} \\
            i &= i + w_{\text{rec}} z_{\text{rec}} \\
            b &= b + \beta z
        \end{align*}

    where :math:`z_{\text{rec}}` and :math:`z_{\text{in}}` are the
    recurrent and input spikes respectively.

    Parameters:
        input_size (int): Size of the input. Also known as the number of input features.
        hidden_size (int): Size of the hidden state. Also known as the number of input features.
        p (LSNNParameters): parameters of the lsnn unit
        dt (float): Integration timestep to use
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: LSNNParameters = LSNNParameters(),
        adjoint: bool = False,
        **kwargs,
    ):
        if adjoint:
            super().__init__(
                activation=lsnn_adjoint_step,
                state_fallback=self.initial_state,
                input_size=input_size,
                hidden_size=hidden_size,
                p=p,
                **kwargs,
            )
        else:
            super().__init__(
                activation=lsnn_step,
                state_fallback=self.initial_state,
                input_size=input_size,
                hidden_size=hidden_size,
                p=p,
                **kwargs,
            )

    def initial_state(
        self,
        input_tensor: torch.Tensor,
    ) -> LSNNState:
        dims = (*input_tensor.shape[:-1], self.hidden_size)
        state = LSNNState(
            z=torch.zeros(
                *dims,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            v=torch.full(
                dims,
                self.p.v_leak.detach(),
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            i=torch.zeros(
                *dims,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            b=self.p.v_th.detach(),
        )
        state.v.requires_grad = True
        return state


class LSNN(SNN):
    r"""A Long short-term memory neuron module *without* recurrence
    adapted from https://arxiv.org/abs/1803.09574

    Usage:
      >>> from norse.torch import LSNN
      >>> layer = LSNN()
      >>> data  = torch.zeros(5, 2)
      >>> output, state = layer.forward(data)

    Parameters:
        p (LSNNParameters): The neuron parameters as a torch Module, which allows the module
            to configure neuron parameters as optimizable.
        dt (float): Time step to use in integration. Defaults to 0.001.
    """

    def __init__(
        self, p: LSNNParameters = LSNNParameters(), adjoint: bool = False, **kwargs
    ):
        if adjoint:
            super().__init__(
                activation=lsnn_feed_forward_adjoint_step,
                state_fallback=self.initial_state,
                p=p,
                **kwargs,
            )
        else:
            super().__init__(
                activation=lsnn_feed_forward_step,
                state_fallback=self.initial_state,
                p=p,
                **kwargs,
            )

    def initial_state(self, input_tensor: torch.Tensor) -> LSNNFeedForwardState:
        state = LSNNFeedForwardState(
            v=torch.full(
                input_tensor.shape[1:],
                self.p.v_leak.detach(),
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            i=torch.zeros(
                *input_tensor.shape[1:],
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            b=torch.full(
                input_tensor.shape[1:],
                self.p.v_th.detach(),
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
        )
        state.v.requires_grad = True
        return state


class LSNNRecurrent(SNNRecurrent):
    r"""A Long short-term memory neuron module *wit* recurrence
    adapted from https://arxiv.org/abs/1803.09574

    Usage:
      >>> from norse.torch.module import LSNNRecurrent
      >>> layer = LSNNRecurrent(2, 10)          // Shape 2 -> 10
      >>> data  = torch.zeros(2, 5, 2)          // Arbitrary data
      >>> output, state = layer.forward(data)   // Out: (2, 5, 10)

    Parameters:
        input_size (int): Size of the input. Also known as the number of input features.
        hidden_size (int): Size of the hidden state. Also known as the number of input features.
        p (LSNNParameters): The neuron parameters as a torch Module, which allows the module
            to configure neuron parameters as optimizable.
        dt (float): Time step to use in integration. Defaults to 0.001.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: LSNNParameters = LSNNParameters(),
        adjoint: bool = False,
        **kwargs,
    ):
        if adjoint:
            super().__init__(
                activation=lsnn_adjoint_step,
                state_fallback=self.initial_state,
                input_size=input_size,
                hidden_size=hidden_size,
                p=p,
                **kwargs,
            )
        else:
            super().__init__(
                activation=lsnn_step,
                state_fallback=self.initial_state,
                input_size=input_size,
                hidden_size=hidden_size,
                p=p,
                **kwargs,
            )

    def initial_state(self, input_tensor: torch.Tensor) -> LSNNState:
        dims = (*input_tensor.shape[1:-1], self.hidden_size)
        state = LSNNState(
            z=torch.zeros(
                *dims,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            v=torch.full(
                dims,
                self.p.v_leak.detach(),
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            i=torch.zeros(
                *dims,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            b=torch.full(
                dims,
                self.p.v_th.detach(),
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
        )
        state.v.requires_grad = True
        return state
