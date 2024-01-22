"""A simplified version of the popular leaky integrate-and-fire neuron model that combines a :mod:`norse.torch.functional.leaky_integrator` with spike thresholds to produce events (spikes).
Compared to the :mod:`norse.torch.functional.lif` modules, this model leaves out the current term, making it computationally simpler but impossible to implement in physical systems because currents cannot "jump" in nature.
It is these sudden current jumps that gives the model its name, because the shift in current is instantaneous and can be drawn as "current boxes".
"""

import torch
from norse.torch.functional.lif_box import (
    LIFBoxFeedForwardState,
    LIFBoxParameters,
    lif_box_feed_forward_step,
)
from norse.torch.module.snn import SNNCell
from norse.torch.utils.clone import clone_tensor


class LIFBoxCell(SNNCell):
    r"""Computes a single euler-integration step for a lif neuron-model without
    current terms.
    It takes as input the input current as generated by an arbitrary torch
    module or function. More specifically it implements one integration
    step of the following ODE

    .. math::
        \dot{v} = 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i)

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}})

    and transition equations

    .. math::
        v = (1-z) v + z v_{\text{reset}}

    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        state (LIFBoxFeedForwardState): current state of the LIF neuron
        p (LIFBoxParameters): parameters of a leaky integrate and fire neuron
        dt (float): Integration timestep to use
    """

    def __init__(self, p: LIFBoxParameters = LIFBoxParameters(), dt: float = 0.001):
        super().__init__(lif_box_feed_forward_step, self.initial_state, p, dt=dt)

    def initial_state(self, input_tensor: torch.Tensor) -> LIFBoxFeedForwardState:
        state = LIFBoxFeedForwardState(v=clone_tensor(self.p.v_leak))
        state.v.requires_grad = True
        return state
