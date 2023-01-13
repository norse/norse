import torch

from norse.torch.functional.lif_adex import (
    LIFAdExState,
    LIFAdExFeedForwardState,
)

from norse.torch.functional.lif_adex_refrac import (
    LIFAdExRefracParameters,
    LIFAdExRefracState,
    LIFAdExRefracFeedForwardState,
    lif_adex_refrac_step,
    lif_adex_refrac_feed_forward_step,
)

from norse.torch.module.snn import SNNCell, SNNRecurrentCell, SNNRecurrent


class LIFAdExRefracCell(SNNCell):
    r"""Module that computes a single euler-integration step of a
    LIFAdEx neuron-model with absolute refractory period *without* recurrence.
    More specifically it implements one integration step of the following ODE.

        .. math::
            \begin{align*}
                \dot{v} &= (1/\tau_{\text{mem}} \left(v_{\text{leak}} - v + i + \Delta_T exp\left({{v - v_{\text{th}}} \over {\Delta_T}}\right)\right))(1-\Theta(\rho)) \\
                \dot{i} &= -1/\tau_{\text{syn}} i \\
                \dot{a} &= 1/\tau_{\text{ada}} \left( a_{current} (V - v_{\text{leak}}) - a \right)\\
                \dot{\rho} &= -1/\tau_{\text{refrac}} \Theta(\rho)
            \end{align*}

        together with the jump condition

        .. math::
            z = \Theta(v - v_{\text{th}})

        and transition equations

        .. math::
            i = i + i_{\text{in}}

        where :math:`i_{\text{in}}` is meant to be the result of applying
        an arbitrary pytorch module (such as a convolution) to input spikes.

        Parameters:
            p (LIFAdExRefracParameters): Parameters of the LIFEx with absolute refractory period neuron model.
            dt (float): Time step to use.

        Examples:
            >>> batch_size = 16
            >>> lif_ex = LIFAdExCell()
            >>> data = torch.randn(batch_size, 20, 30)
            >>> output, s0 = lif_ex(data)
    """

    def __init__(
        self, p: LIFAdExRefracParameters = LIFAdExRefracParameters(), **kwargs
    ) -> None:
        super().__init__(
            lif_adex_refrac_feed_forward_step,
            self.initial_state,
            p=p,
            **kwargs,
        )

    def initial_state(
        self, input_tensor: torch.Tensor
    ) -> LIFAdExRefracFeedForwardState:
        state = LIFAdExRefracFeedForwardState(
            LIFAdExFeedForwardState(
                v=self.p.lif_adex.v_leak.detach(),
                i=torch.zeros(
                    *input_tensor.shape,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                a=torch.zeros(
                    *input_tensor.shape,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
            ),
            rho=torch.zeros(
                input_tensor.shape,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
        )
        state.lif_adex.v.requires_grad = True
        return state


class LIFAdExRefracRecurrentCell(SNNRecurrentCell):
    r"""Module that computes a single euler-integration step of a
    LIFAdEx neuron-model with absolute refractory period
    More specifically it implements one integration step of the following ODE.

        .. math::
            \begin{align*}
                \dot{v} &= (1/\tau_{\text{mem}} \left(v_{\text{leak}} - v + i + \Delta_T exp\left({{v - v_{\text{th}}} \over {\Delta_T}}\right)\right))(1-\Theta(\rho)) \\
                \dot{i} &= -1/\tau_{\text{syn}} i \\
                \dot{a} &= 1/\tau_{\text{ada}} \left( a_{current} (V - v_{\text{leak}}) - a \right)\\
                \dot{\rho} &= -1/\tau_{\text{refrac}} \Theta(\rho)
            \end{align*}

        together with the jump condition

        .. math::
            z = \Theta(v - v_{\text{th}})

        and transition equations

        .. math::
            i = i + i_{\text{in}}

        where :math:`i_{\text{in}}` is meant to be the result of applying
        an arbitrary pytorch module (such as a convolution) to input spikes.

        Parameters:
            p (LIFAdExRefracParameters): Parameters of the LIFEx with absolute refractory period neuron model.
            dt (float): Time step to use.

        Examples:
            >>> batch_size = 16
            >>> lif_ex = LIFAdExRefracCell()
            >>> data = torch.randn(batch_size, 20, 30)
            >>> output, s0 = lif_ex(data)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: LIFAdExRefracParameters = LIFAdExRefracParameters(),
        **kwargs,
    ) -> None:
        super().__init__(
            activation=lif_adex_refrac_step,
            state_fallback=self.initial_state,
            p=p,
            input_size=input_size,
            hidden_size=hidden_size,
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIFAdExRefracState:
        dims = (*input_tensor.shape[:-1], self.hidden_size)
        state = LIFAdExRefracState(
            LIFAdExState(
                z=torch.zeros(
                    *dims,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                v=torch.full(
                    dims,
                    self.p.lif_adex.v_leak.detach(),
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                i=torch.zeros(
                    *dims,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                a=torch.zeros(
                    *dims,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
            ),
            rho=torch.zeros(
                *dims,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
        )
        state.lif_adex.v.requires_grad = True
        return state


class LIFAdExRefracRecurrent(SNNRecurrent):
    r"""A neuron layer that wraps a :class:`LIFAdExRefracRecurrentCell` in time such
    that the layer keeps track of temporal sequences of spikes.
    (spikes from all timesteps, state from the last timestep).

    Example:
    >>> data = torch.zeros(10, 5, 2) # 10 timesteps, 5 batches, 2 neurons
    >>> l = LIFAdExRefracRecurrent(2, 4)
    >>> l(data) # Returns tuple of (Tensor(10, 5, 4), LIFAdExRefracState)

    Parameters:
        input_size (int): The number of input neurons
        hidden_size (int): The number of hidden neurons
        p (LIFAdExRefracParameters): The neuron parameters as a torch Module, which allows the module
            to configure neuron parameters as optimizable.
        input_weights (torch.Tensor): Weights used for input tensors. Defaults to a random
            matrix normalized to the number of hidden neurons.
        recurrent_weights (torch.Tensor): Weights used for input tensors. Defaults to a random
            matrix normalized to the number of hidden neurons.
        autapses (bool): Allow self-connections in the recurrence? Defaults to False. Will also
            remove autapses in custom recurrent weights, if set above.
        dt (float): Time step to use in integration. Defaults to 0.001.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: LIFAdExRefracParameters = LIFAdExRefracParameters(),
        **kwargs,
    ):
        super().__init__(
            activation=lif_adex_refrac_step,
            state_fallback=self.initial_state,
            input_size=input_size,
            hidden_size=hidden_size,
            p=p,
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIFAdExRefracState:
        dims = (
            *input_tensor.shape[1:-1],
            self.hidden_size,
        )

        state = LIFAdExRefracState(
            LIFAdExState(
                z=torch.zeros(
                    *dims,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                v=torch.full(
                    dims,
                    self.p.lif_adex.v_leak.detach(),
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                i=torch.zeros(
                    *dims,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                a=torch.zeros(
                    *dims,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
            ),
            rho=torch.zeros(
                *dims, device=input_tensor.device, dtype=input_tensor.dtype
            ),
        )
        state.lif_adex.v.requires_grad = True
        return state
