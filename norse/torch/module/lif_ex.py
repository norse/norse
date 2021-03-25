import torch

from norse.torch.functional.lif_ex import (
    LIFExState,
    LIFExFeedForwardState,
    LIFExParameters,
    lif_ex_step,
    lif_ex_feed_forward_step,
)

from norse.torch.module.snn import SNNCell, SNNRecurrentCell, SNN, SNNRecurrent


class LIFExCell(SNNCell):
    r"""Computes a single euler-integration step of a recurrent
    exponential LIF neuron-model (*without* recurrence) adapted from
    https://neuronaldynamics.epfl.ch/online/Ch5.S2.html.
    More specifically it implements one integration step of the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} \left(v_{\text{leak}} - v + i + \Delta_T exp\left({{v - v_{\text{th}}} \over {\Delta_T}}\right)\right) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}})

    and transition equations

    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\\text{reset}}
        \end{align*}

    where :math:`z_{\text{rec}}` and :math:`z_{\text{in}}` are the
    recurrent and input spikes respectively.

    Parameters:
        input_size (int): Size of the input.
        hidden_size (int): Size of the hidden state.
        p (LIFExParameters): Parameters of the LIF neuron model.
        dt (float): Time step to use.
        autapses (bool): Allow self-connections in the recurrence? Defaults to False.

    Examples:

        >>> batch_size = 16
        >>> lif_ex = LIFExCell(10, 20)
        >>> input = torch.randn(batch_size, 10)
        >>> output, s0 = lif_ex(input)
    """

    def __init__(self, p: LIFExParameters = LIFExParameters(), **kwargs):
        super().__init__(
            lif_ex_feed_forward_step,
            self.initial_state,
            p=p,
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIFExFeedForwardState:
        state = LIFExFeedForwardState(
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
        )
        state.v.requires_grad = True
        return state


class LIFExRecurrentCell(SNNRecurrentCell):
    r"""Computes a single euler-integration step of a recurrent
    exponential LIFEx neuron-model (*with* recurrence) adapted from
    https://neuronaldynamics.epfl.ch/online/Ch5.S2.html.
    More specifically it implements one integration step of the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} \left(v_{\text{leak}} - v + i + \Delta_T exp\left({{v - v_{\text{th}}} \over {\Delta_T}}\right)\right) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}})

    and transition equations

    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\\text{reset}} \\
            i &= i + w_{\text{input}} z_{\text{in}} \\
            i &= i + w_{\text{rec}} z_{\text{rec}}
        \end{align*}

    where :math:`z_{\text{rec}}` and :math:`z_{\text{in}}` are the
    recurrent and input spikes respectively.

    Parameters:
        input_size (int): Size of the input.
        hidden_size (int): Size of the hidden state.
        p (LIFExParameters): Parameters of the LIF neuron model.
        dt (float): Time step to use.
        a (bool): Allow self-connections in the recurrence? Defaults to False.

    Examples:

        >>> batch_size = 16
        >>> lif_ex = LIFExRecurrentCell(10, 20)
        >>> input = torch.randn(batch_size, 10)
        >>> output, s0 = lif_ex(input)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: LIFExParameters = LIFExParameters(),
        **kwargs,
    ):
        super().__init__(
            activation=lif_ex_step,
            state_fallback=self.initial_state,
            p=p,
            input_size=input_size,
            hidden_size=hidden_size,
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIFExState:
        dims = (*input_tensor.shape[:-1], self.hidden_size)
        state = LIFExState(
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
        )
        state.v.requires_grad = True
        return state


class LIFEx(SNN):
    """
    A neuron layer that wraps a :class:`LIFExCell` in time such
    that the layer keeps track of temporal sequences of spikes.
    After application, the layer returns a tuple containing
      (spikes from all timesteps, state from the last timestep).

    Example:
        >>> data = torch.zeros(10, 5, 2) # 10 timesteps, 5 batches, 2 neurons
        >>> l = LIFEx()
        >>> l(data) # Returns tuple of (Tensor(10, 5, 2), LIFExState)

    Parameters:
        p (LIFExParameters): The neuron parameters as a torch Module, which allows the module
            to configure neuron parameters as optimizable. Defaults to None.
        dt (float): Time step to use in integration. Defaults to 0.001.
    """

    def __init__(self, p: LIFExParameters = LIFExParameters(), **kwargs):
        super().__init__(
            activation=lif_ex_feed_forward_step,
            state_fallback=self.initial_state,
            p=p,
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIFExFeedForwardState:
        state = LIFExFeedForwardState(
            v=torch.full(
                input_tensor.shape[1:],  # Assume first dimension is time
                self.p.v_leak.detach(),
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            i=torch.zeros(
                *input_tensor.shape[1:],
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
        )
        state.v.requires_grad = True
        return state


class LIFExRecurrent(SNNRecurrent):
    """
    A neuron layer that wraps a :class:`LIFExRecurrentCell` in time such
    that the layer keeps track of temporal sequences of spikes.
    After application, the module returns a tuple containing
      (spikes from all timesteps, state from the last timestep).

    Example:
        >>> data = torch.zeros(10, 5, 2) # 10 timesteps, 5 batches, 2 neurons
        >>> l = LIFExRecurrent(2, 4)
        >>> l(data) # Returns tuple of (Tensor(10, 5, 4), LIFExState)

    Parameters:
        input_size (int): The number of input neurons
        hidden_size (int): The number of hidden neurons
        p (LIFExParameters): The neuron parameters as a torch Module, which allows the module
            to configure neuron parameters as optimizable. Defaults to None.
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
        p: LIFExParameters = LIFExParameters(),
        **kwargs,
    ):
        super().__init__(
            activation=lif_ex_step,
            state_fallback=self.initial_state,
            input_size=input_size,
            hidden_size=hidden_size,
            p=p,
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIFExState:
        dims = (  # Remove first dimension (time)
            *input_tensor.shape[1:-1],
            self.hidden_size,
        )
        state = LIFExState(
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
        )
        state.v.requires_grad = True
        return state
