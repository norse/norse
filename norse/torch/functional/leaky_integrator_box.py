r"""
Leaky integrators describe a *leaky* neuron membrane that integrates
incoming currents over time, but never spikes. In other words, the
neuron adds up incoming input current, while leaking out some of it
in every timestep.

.. math::
    \begin{align*}
        \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i)
    \end{align*}

This module differs from the `leaky integrator`_ module by excluding the current term,
so any inputs will have immediate, stepwise, impacts on the voltage (hence the "box").

More information can be found on
`Wikipedia <https://en.wikipedia.org/wiki/Biological_neuron_model#Leaky_integrate-and-fire>`_
or in the book `*Neuron Dynamics* by W. Gerstner et al.,
freely available online <https://neuronaldynamics.epfl.ch/online/Ch5.html>`_.
"""

import torch
import torch.jit

from typing import NamedTuple, Tuple


class LIBoxState(NamedTuple):
    """State of a leaky-integrator without a current term

    Parameters:
        v (torch.Tensor): membrane voltage
    """

    v: torch.Tensor


class LIBoxParameters(NamedTuple):
    """Parameters of a leaky integrator without a current term

    Parameters:
        tau_mem_inv (torch.Tensor): inverse membrane time constant
        v_leak (torch.Tensor): leak potential
    """

    tau_mem_inv: torch.Tensor = torch.as_tensor(1.0 / 1e-2)
    v_leak: torch.Tensor = torch.as_tensor(0.0)


def li_box_step(
    input_spikes: torch.Tensor,
    state: LIBoxState,
    input_weights: torch.Tensor,
    p: LIBoxParameters = LIBoxParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIBoxState]:
    r"""Single euler integration step of a leaky-integrator without a current term.
    More specifically it implements a discretized version of the ODE

    .. math::

        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i)
        \end{align*}

    Parameters:
        input_spikes (torch.Tensor); Input spikes
        s (LIState): state of the leaky integrator
        input_weights (torch.Tensor): weights for incoming spikes
        p (LIParameters): parameters of the leaky integrator
        dt (float): integration timestep to use
    """
    # compute current jumps
    i_jump = torch.nn.functional.linear(input_spikes, input_weights)

    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + i_jump)
    v_new = state.v + dv

    return v_new, LIBoxState(v_new)


def li_box_feed_forward_step(
    input_tensor: torch.Tensor,
    state: LIBoxState,
    p: LIBoxParameters = LIBoxParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIBoxState]:
    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + input_tensor)
    v_new = state.v + dv

    return v_new, LIBoxState(v_new)
