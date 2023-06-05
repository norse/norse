r"""
Leaky integrators describe a *leaky* neuron membrane that integrates
incoming currents over time, but never spikes. In other words, the
neuron adds up incoming input current, while leaking out some of it
in every timestep.
This particular "box" module omits the current term, allowing for instantaneous voltage jumps.

See :mod:`norse.torch.functional.leaky_integrator_box` for more information.
"""

import torch

from norse.torch.module.snn import SNN, SNNCell

from ..functional.leaky_integrator_box import (
    li_box_step,
    li_box_feed_forward_step,
    LIBoxState,
    LIBoxParameters,
)


class LIBoxCell(SNNCell):
    r"""
    Leaky integrator cell without current terms.
    More specifically it implements a discretized version of the ODE

    .. math::

        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i)
        \end{align*}

    Parameters:
        p (LIParameters): parameters of the leaky integrator
        dt (float): integration timestep to use
    """

    def __init__(self, p: LIBoxParameters = LIBoxParameters(), **kwargs):
        super().__init__(
            activation=li_box_feed_forward_step,
            state_fallback=self.initial_state,
            p=p,
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIBoxState:
        state = LIBoxState(
            v=self.p.v_leak.detach(),
        )
        state.v.requires_grad = True
        return state
