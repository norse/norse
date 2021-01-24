"""
Base module for spiking neural network (SNN) modules.
"""

from typing import Callable, Optional, Tuple
import torch

from norse.torch.module.util import remove_autopses

FeedforwardActivation = Callable[
    # Input        State         Parameters       dt
    [torch.Tensor, torch.Tensor, torch.nn.Module, float],
    Tuple[torch.Tensor, torch.Tensor],
]

RecurrentActivation = Callable[
    # Input        State         Input weights Rec. weights  Parameters       dt
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.nn.Module, float],
    Tuple[torch.Tensor, torch.Tensor],
]


class FeedforwardSNN(torch.nn.Module):
    """
    Initializes a feed forward SNN *without* time.

    Parameters:
        activation (FeedforwardActivation): The activation function accepting an input tensor, state
            tensor, and parameters module, and returning a tuple of (output spikes, state).
        state_fallback (Callable[[torch.Tensor], torch.Tensor]): A function that can return a
            default state with the correct dimensions, in case no state is provided in the
            forward pass.
        p (torch.nn.Module): The neuron parameters as a torch Module, which allows the module
            to configure neuron parameters as optimizable.
        dt (float): Time step to use in integration. Defaults to 0.001.
    """

    def __init__(
        self,
        activation: FeedforwardActivation,
        state_fallback: Callable[[torch.Tensor], torch.Tensor],
        p: torch.nn.Module,
        dt: float = 0.001,
    ):
        super().__init__()
        self.activation = activation
        self.state_fallback = state_fallback
        self.p = p
        self.dt = dt

    def forward(self, input_tensor: torch.Tensor, state: Optional[torch.Tensor]):
        state = state if state is not None else self.state_fallback(input_tensor)
        return self.activation(input_tensor, state, self.p, self.dt)


class RecurrentSNNCell(torch.nn.Module):
    """
    The base module for recurrent spiking neural networks (RSNN) *without* time.

    Parameters:
        input_size (int): The number of input neurons
        hidden_size (int): The number of hidden neurons
        activation (RecurrentActivation): The activation function accepting an input tensor, state
            tensor, input weights, recurrent weights, and parameters module, and returning a tuple
            of (output spikes (one per timestep), state).
        state_fallback (Callable[[torch.Tensor], torch.Tensor]): A function that can return a
            default state with the correct dimensions, in case no state is provided in the
            forward pass.
        p (torch.nn.Module): The neuron parameters as a torch Module, which allows the module
            to configure neuron parameters as optimizable.
        input_weights (torch.Tensor): Weights used for input tensors. Defaults to a random
            matrix normalized to the number of hidden neurons.
        recurrent_weights (torch.Tensor): Weights used for input tensors. Defaults to a random
            matrix normalized to the number of hidden neurons.
        autopses (bool): Allow self-connections in the recurrence? Defaults to False. Will also
            remove autopses in custom recurrent weights, if set above.
        dt (float): Time step to use in integration. Defaults to 0.001.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation: FeedforwardActivation,
        state_fallback: Callable[[torch.Tensor], torch.Tensor],
        p: torch.nn.Module,
        input_weights: Optional[torch.Tensor] = None,
        recurrent_weights: Optional[torch.Tensor] = None,
        autopses: bool = False,
        dt: float = 0.001,
    ):
        super().__init__()
        self.activation = activation
        self.state_fallback = state_fallback
        self.p = p
        self.dt = dt

        if input_weights is not None:
            self.input_weights = input_weights
        else:
            self.input_weights = torch.nn.Parameter(
                torch.randn(hidden_size, input_size) * torch.sqrt(2 / hidden_size)
            )
        if recurrent_weights is not None:
            self.recurrent_weights = recurrent_weights
        else:
            self.recurrent_weights = torch.randn(hidden_size, hidden_size) * torch.sqrt(
                2 / hidden_size
            )
        self.recurrent_weights = torch.nn.Parameter(
            recurrent_weights if autopses else remove_autopses(recurrent_weights)
        )

    def forward(self, input_tensor: torch.Tensor, state: Optional[torch.Tensor]):
        state = state if state is not None else self.state_fallback(input_tensor)
        return self.activation(
            input_tensor,
            state,
            self.input_weights,
            self.recurrent_weights,
            self.p,
            self.dt,
        )


class RecurrentSNN(torch.nn.Module):
    """
    The base module for recurrent spiking neural networks (RSNN) *with* time.

    Parameters:
        input_size (int): The number of input neurons
        hidden_size (int): The number of hidden neurons
        activation (RecurrentActivation): The activation function accepting an input tensor, state
            tensor, input weights, recurrent weights, and parameters module, and returning a tuple
            of (output spikes (one per timestep), state).
        state_fallback (Callable[[torch.Tensor], torch.Tensor]): A function that can return a
            default state with the correct dimensions, in case no state is provided in the
            forward pass.
        p (torch.nn.Module): The neuron parameters as a torch Module, which allows the module
            to configure neuron parameters as optimizable.
        input_weights (torch.Tensor): Weights used for input tensors. Defaults to a random
            matrix normalized to the number of hidden neurons.
        recurrent_weights (torch.Tensor): Weights used for input tensors. Defaults to a random
            matrix normalized to the number of hidden neurons.
        autopses (bool): Allow self-connections in the recurrence? Defaults to False. Will also
            remove autopses in custom recurrent weights, if set above.
        dt (float): Time step to use in integration. Defaults to 0.001.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation: FeedforwardActivation,
        state_fallback: Callable[[torch.Tensor], torch.Tensor],
        p: torch.nn.Module,
        input_weights: Optional[torch.Tensor] = None,
        recurrent_weights: Optional[torch.Tensor] = None,
        autopses: bool = False,
        dt: float = 0.001,
    ):
        super().__init__()
        self.activation = activation
        self.state_fallback = state_fallback
        self.p = p
        self.dt = dt

        if input_weights is not None:
            self.input_weights = input_weights
        else:
            self.input_weights = torch.nn.Parameter(
                torch.randn(hidden_size, input_size) * torch.sqrt(2 / hidden_size)
            )
        if recurrent_weights is not None:
            self.recurrent_weights = recurrent_weights
        else:
            self.recurrent_weights = torch.randn(hidden_size, hidden_size) * torch.sqrt(
                2 / hidden_size
            )
        self.recurrent_weights = torch.nn.Parameter(
            recurrent_weights if autopses else remove_autopses(recurrent_weights)
        )

    def forward(self, input_tensor: torch.Tensor, state: torch.Tensor):
        state = state if state is not None else self.state_fallback(input_tensor)

        T = input_tensor.shape[0]
        outputs = []

        for ts in range(T):
            out, state = self.activation(
                input_tensor[ts],
                state,
                self.input_weights,
                self.recurrent_weights,
                self.p,
                self.dt,
            )
            outputs += [out]

        return torch.stack(outputs)
