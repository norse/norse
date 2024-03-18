import torch

from norse.torch.functional.lif import LIFState, LIFFeedForwardState, LIFParameters

from norse.torch.functional.lif_refrac import (
    LIFRefracParameters,
    LIFRefracState,
    LIFRefracFeedForwardState,
    lif_refrac_step,
    lif_refrac_feed_forward_step,
    lif_refrac_step_sparse,
)
from norse.torch.functional.adjoint.lif_refrac_adjoint import (
    lif_refrac_adjoint_step,
    lif_refrac_adjoint_step_sparse,
    lif_refrac_feed_forward_adjoint_step,
)
from norse.torch.module.snn import SNNCell, SNNRecurrentCell, SNNRecurrent


class LIFRefracCell(SNNCell):
    r"""Module that computes a single euler-integration step of a
    LIF neuron-model with absolute refractory period *without* recurrence.
    More specifically it implements one integration step of the following ODE.

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (1-\Theta(\rho)) \
            (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i \\
            \dot{\rho} &= -1/\tau_{\text{refrac}} \Theta(\rho)
        \end{align*}

    together with the jump condition

    .. math::
        \begin{align*}
            z &= \Theta(v - v_{\text{th}}) \\
            z_r &= \Theta(-\rho)
        \end{align*}

    and transition equations

    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\text{reset}} \\
            \rho &= \rho + z_r \rho_{\text{reset}}
        \end{align*}

    Parameters:
        p (LIFRefracParameters): parameters of the lif neuron
        dt (float): Integration timestep to use

    Examples:
        >>> batch_size = 16
        >>> lif = LIFRefracCell()
        >>> input = torch.randn(batch_size, 20, 30)
        >>> output, s0 = lif(input)
    """

    def __init__(self, p: LIFRefracParameters = LIFRefracParameters(), **kwargs):
        super().__init__(
            (
                lif_refrac_feed_forward_adjoint_step
                if p.lif.method == "adjoint"
                else lif_refrac_feed_forward_step
            ),
            self.initial_state,
            p=p,
            **kwargs,
        )

    def initial_state(
        self,
        input_tensor: torch.Tensor,
    ) -> LIFRefracFeedForwardState:
        state = LIFRefracFeedForwardState(
            LIFFeedForwardState(
                v=torch.full(
                    input_tensor.shape,
                    self.p.lif.v_leak.detach(),
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                i=torch.zeros(
                    input_tensor.shape,
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
        state.lif.v.requires_grad = True
        return state


class LIFRefracRecurrentCell(SNNRecurrentCell):
    r"""Module that computes a single euler-integration step of a LIF
    neuron-model with absolute refractory period. More specifically it
    implements one integration step of the following ODE.

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (1-\Theta(\rho)) \
            (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i \\
            \dot{\rho} &= -1/\tau_{\text{refrac}} \Theta(\rho)
        \end{align*}

    together with the jump condition

    .. math::
        \begin{align*}
            z &= \Theta(v - v_{\text{th}}) \\
            z_r &= \Theta(-\rho)
        \end{align*}

    and transition equations

    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\text{reset}} \\
            i &= i + w_{\text{input}} z_{\text{in}} \\
            i &= i + w_{\text{rec}} z_{\text{rec}} \\
            \rho &= \rho + z_r \rho_{\text{reset}}
        \end{align*}

    where :math:`z_{\text{rec}}` and :math:`z_{\text{in}}` are the
    recurrent and input spikes respectively.

    Parameters:
        input_size (int): Size of the input. Also known as the number of input features.
        hidden_size (int): Size of the hidden state. Also known as the number of input features.
        p (LIFRefracParameters): parameters of the lif neuron
        dt (float): Integration timestep to use
        autapses (bool): Allow self-connections in the recurrence? Defaults to False.

    Examples:

        >>> batch_size = 16
        >>> lif = LIFRefracRecurrentCell(10, 20)
        >>> input = torch.randn(batch_size, 10)
        >>> output, s0 = lif(input)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: LIFRefracParameters = LIFRefracParameters(),
        **kwargs,
    ):
        super().__init__(
            activation=(
                lif_refrac_adjoint_step
                if p.lif.method == "adjoint"
                else lif_refrac_step
            ),
            state_fallback=self.initial_state,
            input_size=input_size,
            hidden_size=hidden_size,
            p=p,
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIFRefracState:
        state = LIFRefracState(
            LIFState(
                z=torch.zeros(
                    input_tensor.shape[0],
                    self.hidden_size,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                v=self.p.lif.v_leak.detach(),
                i=torch.zeros(
                    input_tensor.shape[0],
                    self.hidden_size,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
            ),
            rho=torch.zeros(
                input_tensor.shape[0],
                self.hidden_size,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
        )
        state.lif.v.requires_grad = True
        return state


class LIFRefracRecurrent(SNNRecurrent):
    """
    A neuron layer that wraps a :class:`LIFRefracRecurrentCell` in time such
    that the layer keeps track of temporal sequences of spikes.
    (spikes from all timesteps, state from the last timestep).

    Example:
    >>> data = torch.zeros(10, 5, 2) # 10 timesteps, 5 batches, 2 neurons
    >>> l = LIFRefracRecurrent(2, 4)
    >>> l(data) # Returns tuple of (Tensor(10, 5, 4), LIFRefracState)

    Parameters:
        input_size (int): The number of input neurons
        hidden_size (int): The number of hidden neurons
        p (LIFRefracParameters): The neuron parameters as a torch Module, which allows the module
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
        p: LIFRefracParameters = LIFRefracParameters(),
        **kwargs,
    ):
        super().__init__(
            activation=(
                lif_refrac_adjoint_step
                if p.lif.method == "adjoint"
                else lif_refrac_step
            ),
            activation_sparse=(
                lif_refrac_adjoint_step_sparse
                if p.lif.method == "adjoint"
                else lif_refrac_step_sparse
            ),
            state_fallback=self.initial_state,
            input_size=input_size,
            hidden_size=hidden_size,
            p=LIFRefracParameters(
                LIFParameters(
                    torch.as_tensor(p.lif.tau_syn_inv),
                    torch.as_tensor(p.lif.tau_mem_inv),
                    torch.as_tensor(p.lif.v_leak),
                    torch.as_tensor(p.lif.v_th),
                    torch.as_tensor(p.lif.v_reset),
                    p.lif.method,
                    torch.as_tensor(p.lif.alpha),
                ),
                torch.as_tensor(p.rho_reset),
            ),
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIFRefracState:
        dims = (*input_tensor.shape[1:-1], self.hidden_size)
        lif_state = LIFState(
            z=(
                torch.zeros(
                    *dims, device=input_tensor.device, dtype=input_tensor.dtype
                ).to_sparse()
                if input_tensor.is_sparse
                else torch.zeros(
                    *dims, device=input_tensor.device, dtype=input_tensor.dtype
                )
            ),
            v=torch.full(
                dims,
                torch.as_tensor(self.p.lif.v_leak).detach(),
                device=input_tensor.device,
                dtype=torch.float32,
            ),
            i=torch.zeros(
                *dims,
                device=input_tensor.device,
                dtype=torch.float32,
            ),
        )
        state = LIFRefracState(
            lif_state,
            torch.zeros(*dims, device=input_tensor.device, dtype=torch.float32),
        )
        state.lif.v.requires_grad = True
        return state
