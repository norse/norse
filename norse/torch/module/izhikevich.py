import torch

from ..functional.izhikevich import (
    IzhikevichState,
    IzhikevichRecurrentState,
    IzhikevichSpikingBehavior,
    izhikevich_feed_forward_step,
    izhikevich_recurrent_step,
)

from norse.torch.module.snn import SNNCell, SNN, SNNRecurrent, SNNRecurrentCell


def _convert_to_recurrent_state(state: IzhikevichState) -> IzhikevichRecurrentState:
    z = torch.as_tensor(0)
    rstate = IzhikevichRecurrentState(z=z, v=state.v, u=state.u)
    return rstate


class IzhikevichCell(SNNCell):
    """
    Module that computes a single Izhikevich neuron-model *without* recurrence and *without* time.
    More specifically it implements one integration step of the following ODE:

    .. math::
        \\begin{align*}
            \\dot{v} &= 0.04v² + 5v + 140 - u + I
            \\dot{u} &= a(bv - u)
        \\end{align*}

    and

    .. math::
        \\text{if} v = 30 \\text{mV, then} v = c \\text{and} u = u + d

    Example with tonic spiking:
    >>> from norse.torch import IzhikevichCell, tonic_spiking
    >>> batch_size = 16
    >>> cell = IzhikevichCell(tonic_spiking)
    >>> input = torch.randn(batch_size, 10)
    >>> output, s0 = cell(input)

    Parameters:
        spiking_method (IzhikevichSpikingBehavior) : parameters and initial state of the neuron
    """

    def __init__(self, spiking_method: IzhikevichSpikingBehavior, **kwargs):
        super().__init__(
            izhikevich_feed_forward_step, self.initial_state, spiking_method.p, **kwargs
        )
        self.spiking_method = spiking_method

    def initial_state(self, input_tensor: torch.Tensor) -> IzhikevichState:
        state = self.spiking_method.s
        state.v.requires_grad = True
        return state


class IzhikevichRecurrentCell(SNNRecurrentCell):
    """Module that computes a single euler-integration step of an Izhikevich neuron-model *with* recurrence but *without* time.
    More specifically it implements one integration step of the following ODE :

    .. math::
        \\begin{align*}
            \\dot{v} &= 0.04v² + 5v + 140 - u + I
            \\dot{u} &= a(bv - u)
        \\end{align*}

    and

    .. math::
        \\text{if} v = 30 \\text{mV, then} v = c \\text{and} u = u + d

    Example with tonic spiking:
    >>> from norse.torch import IzhikevichRecurrentCell, tonic_spiking
    >>> batch_size = 16
    >>> data = torch.zeros(5, 2) # 5 batches, 2 neurons
    >>> l = IzhikevichRecurrentCell(2, 4)
    >>> l(data) # Returns tuple of (Tensor(5, 4), IzhikevichState)

    Parameters:
        input_size (int): Size of the input. Also known as the number of input features. Defaults to None
        hidden_size (int): Size of the hidden state. Also known as the number of input features. Defaults to None
        p (IzhikevichParameters): Parameters of the Izhikevich neuron model.
        input_weights (torch.Tensor): Weights used for input tensors. Defaults to a random
            matrix normalized to the number of hidden neurons.
        recurrent_weights (torch.Tensor): Weights used for input tensors. Defaults to a random
            matrix normalized to the number of hidden neurons.
        autapses (bool): Allow self-connections in the recurrence? Defaults to False. Will also
            remove autapses in custom recurrent weights, if set above.
        dt (float): Time step to use. Defaults to 0.001.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        spiking_method: IzhikevichSpikingBehavior,
        **kwargs
    ):
        super().__init__(
            activation=izhikevich_recurrent_step,
            state_fallback=self.initial_state,
            p=spiking_method.p,
            input_size=input_size,
            hidden_size=hidden_size,
            **kwargs,
        )
        self.spiking_method = spiking_method

    def initial_state(self, input_tensor: torch.Tensor) -> IzhikevichRecurrentState:
        dims = (*input_tensor.shape[:-1], self.hidden_size)
        rstate = _convert_to_recurrent_state(self.spiking_method.s)
        state = IzhikevichRecurrentState(
            z=torch.zeros(
                *dims,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            v=torch.full(
                dims,
                rstate.v.detach(),
                device=input_tensor.device,
                dtype=input_tensor.dtype,
                requires_grad=True,
            ),
            u=torch.full(
                dims,
                rstate.u.detach(),
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
        )
        state.v.requires_grad = True
        return state


class Izhikevich(SNN):
    """
    A neuron layer that wraps a :class:`IzhikevichCell` in time such
    that the layer keeps track of temporal sequences of spikes.
    After application, the layer returns a tuple containing
      (spikes from all timesteps, state from the last timestep).

    Example:
        >>> data = torch.zeros(10, 5, 2) # 10 timesteps, 5 batches, 2 neurons
        >>> l = Izhikevich()
        >>> l(data) # Returns tuple of (Tensor(10, 5, 2), IzhikevichState)

    Parameters:
        p (IzhikevichParameters): The neuron parameters as a torch Module, which allows the module
            to configure neuron parameters as optimizable.
        dt (float): Time step to use in integration. Defaults to 0.001.
    """

    def __init__(self, spiking_method: IzhikevichSpikingBehavior, **kwargs):
        super().__init__(
            activation=izhikevich_feed_forward_step,
            state_fallback=self.initial_state,
            p=spiking_method.p,
            **kwargs,
        )
        self.spiking_method = spiking_method

    def initial_state(self, input_tensor: torch.Tensor) -> IzhikevichState:
        state = self.spiking_method.s
        return state


class IzhikevichRecurrent(SNNRecurrent):
    """
    A neuron layer that wraps a :class:`IzhikevichRecurrentCell` in time such
    that the layer keeps track of temporal sequences of spikes.
    After application, the layer returns a tuple containing
      (spikes from all timesteps, state from the last timestep).

    Example:
    >>> data = torch.zeros(10, 5, 2) # 10 timesteps, 5 batches, 2 neurons
    >>> l = Izhikevich()
    >>> l(data) # Returns tuple of (Tensor(10, 5, 2), IzhikevichState)

    Parameters:
        p (IzhikevichParameters): The neuron parameters as a torch Module, which allows the module
            to configure neuron parameters as optimizable.
        dt (float): Time step to use in integration. Defaults to 0.001.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        spiking_method: IzhikevichSpikingBehavior,
        *args,
        **kwargs
    ):
        super().__init__(
            activation=izhikevich_recurrent_step,
            state_fallback=self.initial_state,
            input_size=input_size,
            hidden_size=hidden_size,
            p=spiking_method.p,
            *args,
            **kwargs,
        )
        self.spiking_method = spiking_method

    def initial_state(self, input_tensor: torch.Tensor) -> IzhikevichRecurrentState:
        dims = (*input_tensor.shape[1:-1], self.hidden_size)
        rstate = _convert_to_recurrent_state(self.spiking_method.s)
        state = IzhikevichRecurrentState(
            z=torch.zeros(
                *dims,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            v=torch.full(
                dims,
                rstate.v.detach(),
                device=input_tensor.device,
                dtype=input_tensor.dtype,
                requires_grad=True,
            ),
            u=torch.full(
                dims,
                rstate.u.detach(),
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
        )
        return state
