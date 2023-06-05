r"""
Leaky integrators describe a *leaky* neuron membrane that integrates
incoming currents over time, but never spikes. In other words, the
neuron adds up incoming input current, while leaking out some of it
in every timestep.

.. math::
    \begin{align*}
        \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
        \dot{i} &= -1/\tau_{\text{syn}} i
    \end{align*}

The first equation describes how the membrane voltage (:math:`v`, across
the membrane) changes over time. A constant amount of current is *leaked*
out every timestep (:math:`v_{\text{leak}}`), while the current
(:math:`i`) is added.

The second equation describes how the current flowing into the neuron
changes in every timestep.

Notice that both equations are parameterized by the *time constant*
:math:`\tau`. This constant controls how *fast* the changes in voltage
and current occurs. A large time constant means a small change.
In Norse, we call this parameter the *inverse* to avoid having to
recalculate the inverse (:math:`\tau_{\text{mem_inv}}` and
:math:`\tau_{\text{syn_inv}}` respectively).
So, for Norse a large inverse time constant means *rapid* changes while
a small inverse time constant means *slow* changes.

Recall that *voltage* is the difference in charge between two points (in
this case the neuron membrane) and *current* is the rate of change or the
amount of current being added/subtracted at each timestep.

More information can be found on
`Wikipedia <https://en.wikipedia.org/wiki/Biological_neuron_model#Leaky_integrate-and-fire>`_
or in the book `*Neuron Dynamics* by W. Gerstner et al.,
freely available online <https://neuronaldynamics.epfl.ch/online/Ch5.html>`_.
"""

import torch
import torch.jit

from typing import NamedTuple, Tuple


class LIState(NamedTuple):
    """State of a leaky-integrator

    Parameters:
        v (torch.Tensor): membrane voltage
        i (torch.Tensor): input current
    """

    v: torch.Tensor
    i: torch.Tensor


class LIParameters(NamedTuple):
    """Parameters of a leaky integrator

    Parameters:
        tau_syn_inv (torch.Tensor): inverse synaptic time constant
        tau_mem_inv (torch.Tensor): inverse membrane time constant
        v_leak (torch.Tensor): leak potential
    """

    tau_syn_inv: torch.Tensor = torch.as_tensor(1.0 / 5e-3)
    tau_mem_inv: torch.Tensor = torch.as_tensor(1.0 / 1e-2)
    v_leak: torch.Tensor = torch.as_tensor(0.0)


def li_step(
    input_spikes: torch.Tensor,
    state: LIState,
    input_weights: torch.Tensor,
    p: LIParameters = LIParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIState]:
    r"""Single euler integration step of a leaky-integrator.
    More specifically it implements a discretized version of the ODE

    .. math::

        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}


    and transition equations

    .. math::
        i = i + w i_{\text{in}}

    Parameters:
        input_spikes (torch.Tensor); Input spikes
        s (LIState): state of the leaky integrator
        input_weights (torch.Tensor): weights for incoming spikes
        p (LIParameters): parameters of the leaky integrator
        dt (float): integration timestep to use
    """
    # compute current jumps
    i_jump = state.i + torch.nn.functional.linear(input_spikes, input_weights)

    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + i_jump)
    v_new = state.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * i_jump
    i_decayed = i_jump + di

    return v_new, LIState(v_new, i_decayed)


def li_feed_forward_step(
    input_tensor: torch.Tensor,
    state: LIState,
    p: LIParameters = LIParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIState]:
    # compute current jumps
    i_jump = state.i + input_tensor
    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + i_jump)
    v_new = state.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * i_jump
    i_decayed = i_jump + di

    return v_new, LIState(v_new, i_decayed)
