"""A current-based leaky integrate-and-fire neuron model, following the
CubaLIF primitive of the Neuromorphic Intermediate Representation (NIR).

See :mod:`norse.torch.functional.cuba_lif` for more information.
"""

import torch
from norse.torch.functional.cuba_lif import (
    CubaLIFFeedForwardState,
    CubaLIFParameters,
    cuba_lif_feed_forward_step,
)
from norse.torch.module.snn import SNN, SNNCell
from norse.torch.utils.clone import clone_tensor


class CubaLIF(SNN):
    r"""Computes N euler-integration steps for a current-based lif neuron-model,
    where N is the outer (time) dimension of the input tensor.
    More specifically it implements one integration step of the following ODE

    .. math::
        \begin{align*}
            \tau_{\text{syn}} \dot{i} &= -i + w_{\text{in}} S \\
            \tau_{\text{mem}} \dot{v} &= (v_{\text{leak}} - v) + R i
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}})

    and transition equations

    .. math::
        v = (1-z) v + z v_{\text{reset}}

    Parameters:
        p (CubaLIFParameters): parameters of the current-based lif neuron
        dt (float): Integration timestep to use
    """

    def __init__(self, p: CubaLIFParameters = CubaLIFParameters(), dt: float = 0.001):
        super().__init__(cuba_lif_feed_forward_step, self.initial_state, p, dt=dt)

    def initial_state(self, input_tensor: torch.Tensor) -> CubaLIFFeedForwardState:
        state = CubaLIFFeedForwardState(
            v=clone_tensor(self.p.v_leak),
            i=torch.zeros_like(self.p.v_leak),
        )
        state.v.requires_grad = True
        return state


class CubaLIFCell(SNNCell):
    r"""Computes a single euler-integration step for a current-based lif
    neuron-model. More specifically it implements one integration step of
    the following ODE

    .. math::
        \begin{align*}
            \tau_{\text{syn}} \dot{i} &= -i + w_{\text{in}} S \\
            \tau_{\text{mem}} \dot{v} &= (v_{\text{leak}} - v) + R i
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}})

    and transition equations

    .. math::
        v = (1-z) v + z v_{\text{reset}}

    Parameters:
        p (CubaLIFParameters): parameters of the current-based lif neuron
        dt (float): Integration timestep to use
    """

    def __init__(self, p: CubaLIFParameters = CubaLIFParameters(), dt: float = 0.001):
        super().__init__(cuba_lif_feed_forward_step, self.initial_state, p, dt=dt)

    def initial_state(self, input_tensor: torch.Tensor) -> CubaLIFFeedForwardState:
        state = CubaLIFFeedForwardState(
            v=clone_tensor(self.p.v_leak),
            i=torch.zeros_like(self.p.v_leak),
        )
        state.v.requires_grad = True
        return state
