"""
A very popular neuron model that combines a :mod:`norse.torch.module.leaky_integrator` with
spike thresholds to produce events (spikes).

See :mod:`norse.torch.functional.lif` for more information.
"""

import torch

from norse.torch.functional.lif import (
    LIFState,
    LIFFeedForwardState,
    LIFParameters,
    lif_step,
    lif_step_sparse,
    lif_feed_forward_step,
    lif_feed_forward_step_sparse,
)
from norse.torch.functional.adjoint.lif_adjoint import (
    lif_adjoint_step,
    lif_adjoint_step_sparse,
    lif_feed_forward_adjoint_step,
    lif_feed_forward_adjoint_step_sparse,
)
from norse.torch.module.snn import SNN, SNNCell, SNNRecurrent, SNNRecurrentCell
from norse.torch.utils.clone import clone_tensor


class LIFCell(SNNCell):
    """Module that computes a single euler-integration step of a
    leaky integrate-and-fire (LIF) neuron-model *without* recurrence and *without* time.

    More specifically it implements one integration step
    of the following ODE

    .. math::
        \\begin{align*}
            \\dot{v} &= 1/\\tau_{\\text{mem}} (v_{\\text{leak}} - v + i) \\
            \\dot{i} &= -1/\\tau_{\\text{syn}} i
        \\end{align*}

    together with the jump condition

    .. math::
        z = \\Theta(v - v_{\\text{th}})

    and transition equations

    .. math::
        \\begin{align*}
            v &= (1-z) v + z v_{\\text{reset}}
        \\end{align*}

    Example:
        >>> data = torch.zeros(5, 2) # 5 batches, 2 neurons
        >>> l = LIFCell(2, 4)
        >>> l(data) # Returns tuple of (Tensor(5, 4), LIFState)

    Arguments:
        p (LIFParameters): Parameters of the LIF neuron model.
        sparse (bool): Whether to apply sparse activation functions (True) or not (False). Defaults to False.
        dt (float): Time step to use. Defaults to 0.001.
    """

    def __init__(self, p: LIFParameters = LIFParameters(), **kwargs):
        super().__init__(
            activation=(
                lif_feed_forward_adjoint_step
                if p.method == "adjoint"
                else lif_feed_forward_step
            ),
            activation_sparse=(
                lif_feed_forward_adjoint_step_sparse
                if p.method == "adjoint"
                else lif_feed_forward_step_sparse
            ),
            state_fallback=self.initial_state,
            p=LIFParameters(
                torch.as_tensor(p.tau_syn_inv),
                torch.as_tensor(p.tau_mem_inv),
                torch.as_tensor(p.v_leak),
                torch.as_tensor(p.v_th),
                torch.as_tensor(p.v_reset),
                p.method,
                torch.as_tensor(p.alpha),
            ),
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIFFeedForwardState:
        state = LIFFeedForwardState(
            v=clone_tensor(self.p.v_leak),
            i=torch.zeros(
                input_tensor.shape,
                device=input_tensor.device,
                dtype=torch.float32,
            ),
        )
        state.v.requires_grad = True
        return state


class LIFRecurrentCell(SNNRecurrentCell):
    """Module that computes a single euler-integration step of a
    leaky integrate-and-fire (LIF) neuron-model *with* recurrence but *without* time.
    More specifically it implements one integration step
    of the following ODE

    .. math::
        \\begin{align*}
            \\dot{v} &= 1/\\tau_{\\text{mem}} (v_{\\text{leak}} - v + i) \\
            \\dot{i} &= -1/\\tau_{\\text{syn}} i
        \\end{align*}

    together with the jump condition

    .. math::
        z = \\Theta(v - v_{\\text{th}})

    and transition equations

    .. math::
        \\begin{align*}
            v &= (1-z) v + z v_{\\text{reset}} \\
            i &= i + w_{\\text{input}} z_{\\text{in}} \\
            i &= i + w_{\\text{rec}} z_{\\text{rec}}
        \\end{align*}

    where :math:`z_{\\text{rec}}` and :math:`z_{\\text{in}}` are the
    recurrent and input spikes respectively.

    Example:
        >>> data = torch.zeros(5, 2) # 5 batches, 2 neurons
        >>> l = LIFRecurrentCell(2, 4)
        >>> l(data) # Returns tuple of (Tensor(5, 4), LIFState)

    Parameters:
        input_size (int): Size of the input. Also known as the number of input features.
        hidden_size (int): Size of the hidden state. Also known as the number of input features.
        p (LIFParameters): Parameters of the LIF neuron model.
        sparse (bool): Whether to apply sparse activation functions (True) or not (False). Defaults to False.
        input_weights (torch.Tensor): Weights used for input tensors. Defaults to a random
            matrix normalized to the number of hidden neurons.
        recurrent_weights (torch.Tensor): Weights used for input tensors. Defaults to a random
            matrix normalized to the number of hidden neurons.
        autapses (bool): Allow self-connections in the recurrence? Defaults to False. Will also
            remove autapses in custom recurrent weights, if set above.
        dt (float): Time step to use.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: LIFParameters = LIFParameters(),
        **kwargs,
    ):
        super().__init__(
            activation=lif_adjoint_step if p.method == "adjoint" else lif_step,
            activation_sparse=(
                lif_adjoint_step_sparse if p.method == "adjoint" else lif_step_sparse
            ),
            state_fallback=self.initial_state,
            p=LIFParameters(
                torch.as_tensor(p.tau_syn_inv),
                torch.as_tensor(p.tau_mem_inv),
                torch.as_tensor(p.v_leak),
                torch.as_tensor(p.v_th),
                torch.as_tensor(p.v_reset),
                p.method,
                torch.as_tensor(p.alpha),
            ),
            input_size=input_size,
            hidden_size=hidden_size,
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIFState:
        dims = (*input_tensor.shape[:-1], self.hidden_size)
        state = LIFState(
            z=(
                torch.zeros(
                    dims,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ).to_sparse()
                if input_tensor.is_sparse
                else torch.zeros(
                    dims,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                )
            ),
            v=torch.full(
                dims,
                torch.as_tensor(self.p.v_leak).detach(),
                device=input_tensor.device,
                dtype=torch.float32,
            ),
            i=torch.zeros(
                dims,
                device=input_tensor.device,
                dtype=torch.float32,
            ),
        )
        state.v.requires_grad = True
        return state


class LIF(SNN):
    """
    A neuron layer that wraps a :class:`LIFCell` in time such
    that the layer keeps track of temporal sequences of spikes.
    After application, the layer returns a tuple containing
      (spikes from all timesteps, state from the last timestep).

    Example:
        >>> data = torch.zeros(10, 5, 2) # 10 timesteps, 5 batches, 2 neurons
        >>> l = LIF()
        >>> l(data) # Returns tuple of (Tensor(10, 5, 2), LIFState)

    Parameters:
        p (LIFParameters): The neuron parameters as a torch Module, which allows the module
            to configure neuron parameters as optimizable.
        sparse (bool): Whether to apply sparse activation functions (True) or not (False). Defaults to False.
        dt (float): Time step to use in integration. Defaults to 0.001.
    """

    def __init__(self, p: LIFParameters = LIFParameters(), **kwargs):
        super().__init__(
            activation=(
                lif_feed_forward_adjoint_step
                if p.method == "adjoint"
                else lif_feed_forward_step
            ),
            activation_sparse=(
                lif_feed_forward_adjoint_step_sparse
                if p.method == "adjoint"
                else lif_feed_forward_step_sparse
            ),
            state_fallback=self.initial_state,
            p=LIFParameters(
                torch.as_tensor(p.tau_syn_inv),
                torch.as_tensor(p.tau_mem_inv),
                torch.as_tensor(p.v_leak),
                torch.as_tensor(p.v_th),
                torch.as_tensor(p.v_reset),
                p.method,
                torch.as_tensor(p.alpha),
            ),
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIFFeedForwardState:
        state = LIFFeedForwardState(
            v=torch.full(
                input_tensor.shape[1:],  # Assume first dimension is time
                torch.as_tensor(self.p.v_leak).detach(),
                device=input_tensor.device,
                dtype=torch.float32,
            ),
            i=torch.zeros(
                input_tensor.shape[1:],
                device=input_tensor.device,
                dtype=torch.float32,
            ),
        )
        state.v.requires_grad = True
        return state


class LIFRecurrent(SNNRecurrent):
    """
    A neuron layer that wraps a :class:`LIFRecurrentCell` in time such
    that the layer keeps track of temporal sequences of spikes.
    After application, the module returns a tuple containing
      (spikes from all timesteps, state from the last timestep).

    Example:
        >>> data = torch.zeros(10, 5, 2) # 10 timesteps, 5 batches, 2 neurons
        >>> l = LIFRecurrent(2, 4)
        >>> l(data) # Returns tuple of (Tensor(10, 5, 4), LIFState)

    Parameters:
        input_size (int): The number of input neurons
        hidden_size (int): The number of hidden neurons
        p (LIFParameters): The neuron parameters as a torch Module, which allows the module
            to configure neuron parameters as optimizable.
        sparse (bool): Whether to apply sparse activation functions (True) or not (False). Defaults to False.
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
        p: LIFParameters = LIFParameters(),
        **kwargs,
    ):
        super().__init__(
            activation=lif_adjoint_step if p.method == "adjoint" else lif_step,
            activation_sparse=(
                lif_adjoint_step_sparse if p.method == "adjoint" else lif_step_sparse
            ),
            state_fallback=self.initial_state,
            input_size=input_size,
            hidden_size=hidden_size,
            p=LIFParameters(
                torch.as_tensor(p.tau_syn_inv),
                torch.as_tensor(p.tau_mem_inv),
                torch.as_tensor(p.v_leak),
                torch.as_tensor(p.v_th),
                torch.as_tensor(p.v_reset),
                p.method,
                torch.as_tensor(p.alpha),
            ),
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIFState:
        dims = (  # Remove first dimension (time)
            *input_tensor.shape[1:-1],
            self.hidden_size,
        )
        state = LIFState(
            z=(
                torch.zeros(
                    dims,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ).to_sparse()
                if input_tensor.is_sparse
                else torch.zeros(
                    dims,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                )
            ),
            v=torch.full(
                dims,
                torch.as_tensor(self.p.v_leak).detach(),
                device=input_tensor.device,
                dtype=torch.float32,
            ),
            i=torch.zeros(
                dims,
                device=input_tensor.device,
                dtype=torch.float32,
            ),
        )
        state.v.requires_grad = True
        return state
