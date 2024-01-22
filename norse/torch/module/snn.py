"""
Base module for spiking neural network (SNN) modules.
"""

from typing import Any, Callable, List, Optional, Tuple
import torch

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


def _merge_states(states: List[Any]):
    """
    Merges states recursively by using :method:`torch.stack` on individual state variables to
    produce a single output tuple, with an extra outer dimension.

    Arguments:
        states (List[Tuple]): The input list of states to merge

    Return:
        A single state of the same type as the first state in the input list of states, but with
        its members replaced with a stacked version of the members from the input states.
    """
    state_dict = states[0]._asdict()
    cls = states[0].__class__
    keys = list(state_dict.keys())
    tuples = [isinstance(s, tuple) for s in state_dict.values()]
    output_dict = {}
    for key, nested in zip(keys, tuples):
        if nested:
            nested_list = [getattr(s, key) for s in states]
            output_dict[key] = _merge_states(nested_list)
        else:
            values = [getattr(s, key) for s in states]
            output_dict[key] = torch.stack(values)
    return cls(**output_dict)


class SNNCell(torch.nn.Module):
    """
    Initializes a feedforward neuron cell *without* time.

    Parameters:
        activation (FeedforwardActivation): The activation function accepting an input tensor, state
            tensor, and parameters module, and returning a tuple of (output spikes, state).
        state_fallback (Callable[[torch.Tensor], Any]): A function that can return a
            default state with the correct dimensions, in case no state is provided in the
            forward pass.
        p (torch.nn.Module): The neuron parameters as a torch Module, which allows the module
            to configure neuron parameters as optimizable.
        dt (float): Time step to use in integration. Defaults to 0.001.
        activation_sparse (Optional[FeedforwardActivation]): A Sparse activation function - if it exists
            for the neuron model
    """

    def __init__(
        self,
        activation: FeedforwardActivation,
        state_fallback: Callable[[torch.Tensor], torch.Tensor],
        p: Any,
        dt: float = 0.001,
        activation_sparse: Optional[FeedforwardActivation] = None,
    ):
        super().__init__()
        self.activation = activation
        self.activation_sparse = activation_sparse
        self.state_fallback = state_fallback
        self.p = p
        self.dt = dt

    def extra_repr(self) -> str:
        return f"p={self.p}, dt={self.dt}"

    def forward(self, input_tensor: torch.Tensor, state: Optional[Any] = None):
        state = state if state is not None else self.state_fallback(input_tensor)
        if self.activation_sparse is not None and input_tensor.is_sparse:
            return self.activation_sparse(input_tensor, state, self.p, self.dt)
        else:
            return self.activation(input_tensor, state, self.p, self.dt)


class SNNRecurrentCell(torch.nn.Module):
    """
    The base module for recurrent neuron cell *without* time.

    Parameters:
        activation (RecurrentActivation): The activation function accepting an input tensor, state
            tensor, input weights, recurrent weights, and parameters module, and returning a tuple
            of (output spikes (one per timestep), state).
        state_fallback (Callable[[torch.Tensor], Any]): A function that can return a
            default state with the correct dimensions, in case no state is provided in the
            forward pass.
        input_size (int): The number of input neurons
        hidden_size (int): The number of hidden neurons
        p (torch.nn.Module): The neuron parameters as a torch Module, which allows the module
            to configure neuron parameters as optimizable.
        input_weights (torch.Tensor): Weights used for input tensors. Defaults to a random
            matrix normalized to the number of hidden neurons.
        recurrent_weights (torch.Tensor): Weights used for input tensors. Defaults to a random
            matrix normalized to the number of hidden neurons.
        autapses (bool): Allow self-connections in the recurrence? Defaults to False. Will also
            remove autapses in custom recurrent weights, if set above.
        dt (float): Time step to use in integration. Defaults to 0.001.
        activation_sparse (Optional[RecurrentActivation]): A Sparse activation function - if it exists
            for the neuron model
    """

    def __init__(
        self,
        activation: RecurrentActivation,
        state_fallback: Callable[[torch.Tensor], torch.Tensor],
        input_size: int,
        hidden_size: int,
        p: torch.nn.Module,
        input_weights: Optional[torch.Tensor] = None,
        recurrent_weights: Optional[torch.Tensor] = None,
        autapses: bool = False,
        dt: float = 0.001,
        activation_sparse: Optional[RecurrentActivation] = None,
    ):
        super().__init__()
        self.activation = activation
        self.activation_sparse = activation_sparse
        self.autapses = autapses
        self.state_fallback = state_fallback
        self.p = p
        self.dt = dt
        self.input_size = torch.as_tensor(input_size)
        self.hidden_size = torch.as_tensor(hidden_size)

        if input_weights is not None:
            self.input_weights = input_weights
        else:
            self.input_weights = torch.nn.Parameter(
                torch.randn(self.hidden_size, self.input_size)
                * torch.sqrt(2.0 / self.hidden_size)
            )

        if recurrent_weights is not None:
            self.recurrent_weights = recurrent_weights
        else:
            self.recurrent_weights = torch.nn.Parameter(
                torch.randn(self.hidden_size, self.hidden_size)
                * torch.sqrt(2.0 / self.hidden_size)
            )

        if not autapses:
            with torch.no_grad():
                self.recurrent_weights.fill_diagonal_(0.0)

            # Eradicate gradient updates from autapses
            def autapse_hook(gradient):
                return gradient.clone().fill_diagonal_(0.0)

            self.recurrent_weights.requires_grad = True
            self.recurrent_weights.register_hook(autapse_hook)

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            + f"p={self.p}, autapses={self.autapses}, dt={self.dt}"
        )

    def forward(self, input_tensor: torch.Tensor, state: Optional[Any] = None):
        state = state if state is not None else self.state_fallback(input_tensor)
        if self.activation_sparse is not None and input_tensor.is_sparse:
            return self.activation_sparse(
                input_tensor,
                state,
                self.input_weights,
                self.recurrent_weights,
                self.p,
                self.dt,
            )
        else:
            return self.activation(
                input_tensor,
                state,
                self.input_weights,
                self.recurrent_weights,
                self.p,
                self.dt,
            )


class SNN(torch.nn.Module):
    """
    The base module for spiking neural networks (RSNN) *with* time (*without* recurrence).

    Parameters:
        activation (RecurrentActivation): The activation function accepting an input tensor, state
            tensor, input weights, recurrent weights, and parameters module, and returning a tuple
            of (output spikes (one per timestep), state).
        state_fallback (Callable[[torch.Tensor], torch.Tensor]): A function that can return a
            default state with the correct dimensions, in case no state is provided in the
            forward pass.
        p (torch.nn.Module): The neuron parameters as a torch Module, which allows the module
            to configure neuron parameters as optimizable.
        dt (float): Time step to use in integration. Defaults to 0.001.
        activation_sparse (Optional[FeedforwardActivation]): A Sparse activation function - if it exists
            for the neuron model
        record_states (bool): If True, the module will record and return a state object for each timestep
            during simulation (note that this will consume memory). If False (default), we only return the
            final state during simulation.
    """

    def __init__(
        self,
        activation: FeedforwardActivation,
        state_fallback: Callable[[torch.Tensor], torch.Tensor],
        p: Any,
        dt: float = 0.001,
        activation_sparse: Optional[FeedforwardActivation] = None,
        record_states: bool = False,
    ):
        super().__init__()
        self.activation = activation
        self.activation_sparse = activation_sparse
        self.state_fallback = state_fallback
        self.p = p
        self.dt = dt
        self.record_states = record_states

    def extra_repr(self) -> str:
        return f"p={self.p}, dt={self.dt}"

    def forward(self, input_tensor: torch.Tensor, state: Optional[Any] = None):
        state = state if state is not None else self.state_fallback(input_tensor)

        T = input_tensor.shape[0]
        outputs = []
        states = []

        activation = (
            self.activation_sparse
            if self.activation_sparse is not None and input_tensor.is_sparse
            else self.activation
        )

        for ts in range(T):
            out, state = activation(
                input_tensor[ts],
                state,
                self.p,
                self.dt,
            )
            outputs.append(out)
            if self.record_states:
                states.append(state)

        return torch.stack(outputs), (
            state if not self.record_states else _merge_states(states)
        )


class SNNRecurrent(torch.nn.Module):
    """
    The base module for recurrent spiking neural networks (RSNN) *with* time.

    Parameters:
        activation (RecurrentActivation): The activation function accepting an input tensor, state
            tensor, input weights, recurrent weights, and parameters module, and returning a tuple
            of (output spikes (one per timestep), state).
        state_fallback (Callable[[torch.Tensor], torch.Tensor]): A function that can return a
            default state with the correct dimensions, in case no state is provided in the
            forward pass.
        input_size (int): The number of input neurons
        hidden_size (int): The number of hidden neurons
        p (torch.nn.Module): The neuron parameters as a torch Module, which allows the module
            to configure neuron parameters as optimizable.
        input_weights (torch.Tensor): Weights used for input tensors. Defaults to a random
            matrix normalized to the number of hidden neurons.
        recurrent_weights (torch.Tensor): Weights used for input tensors. Defaults to a random
            matrix normalized to the number of hidden neurons.
        autapses (bool): Allow self-connections in the recurrence? Defaults to False. Will also
            remove autapses in custom recurrent weights, if set above.
        dt (float): Time step to use in integration. Defaults to 0.001.
        activation_sparse (Optional[RecurrentActivation]): A Sparse activation function - if it exists
            for the neuron model
        record_states (bool): If True, the module will record and return a state object for each timestep
            during simulation (note that this will consume memory). If False (default), we only return the
            final state during simulation.
    """

    def __init__(
        self,
        activation: RecurrentActivation,
        state_fallback: Callable[[torch.Tensor], torch.Tensor],
        input_size: int,
        hidden_size: int,
        p: Any,
        input_weights: Optional[torch.Tensor] = None,
        recurrent_weights: Optional[torch.Tensor] = None,
        autapses: bool = False,
        dt: float = 0.001,
        activation_sparse: Optional[RecurrentActivation] = None,
        record_states: bool = False,
    ):
        super().__init__()
        self.activation = activation
        self.activation_sparse = activation_sparse
        self.autapses = autapses
        self.state_fallback = state_fallback
        self.p = p
        self.dt = dt
        self.input_size = torch.as_tensor(input_size)
        self.hidden_size = torch.as_tensor(hidden_size)
        self.record_states = record_states

        if input_weights is not None:
            self.input_weights = input_weights
        else:
            self.input_weights = torch.nn.Parameter(
                torch.randn(self.hidden_size, self.input_size)
                * torch.sqrt(2.0 / self.hidden_size)
            )
        if recurrent_weights is not None:
            self.recurrent_weights = recurrent_weights
        else:
            self.recurrent_weights = torch.nn.Parameter(
                torch.randn(self.hidden_size, self.hidden_size)
                * torch.sqrt(2.0 / self.hidden_size)
            )
        if not autapses:
            with torch.no_grad():
                self.recurrent_weights.fill_diagonal_(0.0)

            # Eradicate gradient updates from autapses
            def autapse_hook(gradient):
                return gradient.clone().fill_diagonal_(0.0)

            self.recurrent_weights.requires_grad = True
            self.recurrent_weights.register_hook(autapse_hook)

    def extra_repr(self) -> str:
        return f"input_size={self.input_size}, hidden_size={self.hidden_size}, p={self.p}, autapses={self.autapses}, dt={self.dt}"

    def forward(self, input_tensor: torch.Tensor, state: Optional[Any] = None):
        state = state if state is not None else self.state_fallback(input_tensor)

        T = input_tensor.shape[0]
        outputs = []
        states = []

        activation = (
            self.activation_sparse
            if self.activation_sparse is not None and input_tensor.is_sparse
            else self.activation
        )

        for ts in range(T):
            out, state = activation(
                input_tensor[ts],
                state,
                self.input_weights,
                self.recurrent_weights,
                self.p,
                self.dt,
            )
            outputs.append(out)
            if self.record_states:
                states.append(state)

        return torch.stack(outputs), (
            state if not self.record_states else _merge_states(states)
        )
