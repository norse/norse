import torch

from typing import Optional, Tuple
import numpy as np

from norse.torch.functional.coba_lif import CobaLIFParameters, CobaLIFState
from norse.torch.functional.coba_lif import coba_lif_step


class CobaLIFCell(torch.nn.Module):
    """Module that computes a single euler-integration step of a conductance based
    LIF neuron-model. More specifically it implements one integration step of
    the following ODE

    .. math::
        \\begin{align*}
            \\dot{v} &= 1/c_{\\text{mem}} (g_l (v_{\\text{leak}} - v) \
              + g_e (E_{\\text{rev_e}} - v) + g_i (E_{\\text{rev_i}} - v)) \\\\
            \\dot{g_e} &= -1/\\tau_{\\text{syn}} g_e \\\\
            \\dot{g_i} &= -1/\\tau_{\\text{syn}} g_i
        \\end{align*}

    together with the jump condition

    .. math::
        z = \\Theta(v - v_{\\text{th}})

    and transition equations

    .. math::
        \\begin{align*}
            v &= (1-z) v + z v_{\\text{reset}} \\\\
            g_e &= g_e + \\text{relu}(w_{\\text{input}}) z_{\\text{in}} \\\\
            g_e &= g_e + \\text{relu}(w_{\\text{rec}}) z_{\\text{rec}} \\\\
            g_i &= g_i + \\text{relu}(-w_{\\text{input}}) z_{\\text{in}} \\\\
            g_i &= g_i + \\text{relu}(-w_{\\text{rec}}) z_{\\text{rec}} \\\\
        \\end{align*}

    where :math:`z_{\\text{rec}}` and :math:`z_{\\text{in}}` are the recurrent
    and input spikes respectively.

    Parameters:
        input_size (int): Size of the input.
        hidden_size (int): Size of the hidden state.
        p (LIFParameters): Parameters of the LIF neuron model.
        dt (float): Time step to use.

    Examples:

        >>> batch_size = 16
        >>> lif = CobaLIFCell(10, 20)
        >>> input = torch.randn(batch_size, 10)
        >>> output, s0 = lif(input)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: CobaLIFParameters = CobaLIFParameters(),
        dt: float = 0.001,
    ):
        super(CobaLIFCell, self).__init__()
        self.input_weights = torch.nn.Parameter(
            torch.randn(hidden_size, input_size) / np.sqrt(input_size)
        )
        self.recurrent_weights = torch.nn.Parameter(
            torch.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.p = p
        self.dt = dt

    def forward(
        self, input_tensor: torch.Tensor, state: Optional[CobaLIFState] = None
    ) -> Tuple[torch.Tensor, CobaLIFState]:
        if state is None:
            state = CobaLIFState(
                z=torch.zeros(
                    input_tensor.shape[0],
                    self.hidden_size,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                v=torch.zeros(
                    input_tensor.shape[0],
                    self.hidden_size,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                g_e=torch.zeros(
                    input_tensor.shape[0],
                    self.hidden_size,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
                g_i=torch.zeros(
                    input_tensor.shape[0],
                    self.hidden_size,
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
            )
            state.v.requires_grad = True
        return coba_lif_step(
            input_tensor,
            state,
            self.input_weights,
            self.recurrent_weights,
            p=self.p,
            dt=self.dt,
        )
