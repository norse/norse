r"""
Klein-Gordon wave neuron model.

A neuron whose membrane potential follows second-order oscillatory dynamics
driven by the Klein-Gordon dispersion relation.  Instead of exponential
decay toward a rest potential, the membrane oscillates at a natural
frequency :math:`\chi` (the *mass term*).

The continuous dynamics are:

.. math::
    \ddot{v} = -\chi^2 v + i

which are discretised via the leapfrog (Verlet) scheme:

.. math::
    v[t+1] = 2 v[t] - v_{\text{prev}}[t] + dt^2 (-\chi^2 v[t] + i[t])

together with the jump condition

.. math::
    z = \Theta(v - v_{\text{th}})

and reset:

.. math::
    v = (1-z) v + z\, v_{\text{reset}}

This enables **frequency-selective resonance**: inputs near
:math:`\omega = \chi` are amplified through parametric driving while
off-resonance inputs are suppressed.

The damped variant adds a friction term :math:`-\gamma\dot{v}`, bridging
between the purely oscillatory regime (:math:`\gamma = 0`) and
first-order leaky integration (:math:`\gamma \gg 1`).
"""

from typing import NamedTuple, Tuple

import torch
import torch.jit

from norse.torch.functional.threshold import threshold
import norse.torch.utils.pytree as pytree


class WaveLIFParameters(
    pytree.StateTuple, metaclass=pytree.MultipleInheritanceNamedTupleMeta
):
    r"""Parameters for a Klein-Gordon wave neuron.

    Parameters:
        chi (torch.Tensor): natural oscillation frequency (mass term)
        gamma (torch.Tensor): damping coefficient (0 = undamped)
        v_th (torch.Tensor): threshold potential for spike generation
        v_reset (torch.Tensor): reset potential after spike
        method (str): surrogate gradient method (default: ``"super"``)
        alpha (float): surrogate gradient sharpness
    """

    chi: torch.Tensor = torch.as_tensor(0.5)
    gamma: torch.Tensor = torch.as_tensor(0.0)
    v_th: torch.Tensor = torch.as_tensor(1.0)
    v_reset: torch.Tensor = torch.as_tensor(0.0)
    method: str = "super"
    alpha: float = torch.as_tensor(100.0)


class WaveLIFState(
    pytree.StateTuple, metaclass=pytree.MultipleInheritanceNamedTupleMeta
):
    r"""State of a recurrent Klein-Gordon wave neuron.

    Parameters:
        z (torch.Tensor): recurrent spikes
        v (torch.Tensor): membrane potential
        v_prev (torch.Tensor): previous membrane potential (for leapfrog)
    """

    z: torch.Tensor
    v: torch.Tensor
    v_prev: torch.Tensor


class WaveLIFFeedForwardState(
    pytree.StateTuple, metaclass=pytree.MultipleInheritanceNamedTupleMeta
):
    r"""State of a feed-forward Klein-Gordon wave neuron.

    Parameters:
        v (torch.Tensor): membrane potential
        v_prev (torch.Tensor): previous membrane potential (for leapfrog)
    """

    v: torch.Tensor
    v_prev: torch.Tensor


def wave_lif_step(
    input_spikes: torch.Tensor,
    state: WaveLIFState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: WaveLIFParameters,
    dt: float = 0.001,
) -> Tuple[torch.Tensor, WaveLIFState]:
    r"""Computes a single leapfrog-integration step of a Klein-Gordon wave
    neuron with recurrent connections.

    .. math::
        v_{\text{new}} = \frac{2 v - (1 - \gamma\,dt/2)\,v_{\text{prev}}
        + dt^2 (-\chi^2 v + i)}{1 + \gamma\,dt/2}

    Parameters:
        input_spikes (torch.Tensor): input spikes at the current time step
        state (WaveLIFState): current state of the neuron
        input_weights (torch.Tensor): synaptic weights for incoming spikes
        recurrent_weights (torch.Tensor): synaptic weights for recurrent spikes
        p (WaveLIFParameters): neuron parameters
        dt (float): integration time step
    """
    # compute input current
    i = torch.nn.functional.linear(
        input_spikes, input_weights
    ) + torch.nn.functional.linear(state.z, recurrent_weights)

    # leapfrog with optional damping
    dt2 = dt * dt
    gd2 = p.gamma * dt * 0.5

    v_new = (
        2.0 * state.v
        - (1.0 - gd2) * state.v_prev
        + dt2 * (-p.chi * p.chi * state.v + i)
    ) / (1.0 + gd2)

    # spike generation
    z_new = threshold(v_new - p.v_th, p.method, p.alpha)

    # reset
    v_new = (1 - z_new.detach()) * v_new + z_new.detach() * p.v_reset

    return z_new, WaveLIFState(z=z_new, v=v_new, v_prev=state.v)


def wave_lif_feed_forward_step(
    input_tensor: torch.Tensor,
    state: WaveLIFFeedForwardState,
    p: WaveLIFParameters,
    dt: float = 0.001,
) -> Tuple[torch.Tensor, WaveLIFFeedForwardState]:
    r"""Computes a single leapfrog-integration step of a Klein-Gordon wave
    neuron in feed-forward mode (no recurrence).

    .. math::
        v_{\text{new}} = \frac{2 v - (1 - \gamma\,dt/2)\,v_{\text{prev}}
        + dt^2 (-\chi^2 v + I_{\text{in}})}{1 + \gamma\,dt/2}

    Parameters:
        input_tensor (torch.Tensor): input current at the current time step
        state (WaveLIFFeedForwardState): current state of the neuron
        p (WaveLIFParameters): neuron parameters
        dt (float): integration time step
    """
    dt2 = dt * dt
    gd2 = p.gamma * dt * 0.5

    v_new = (
        2.0 * state.v
        - (1.0 - gd2) * state.v_prev
        + dt2 * (-p.chi * p.chi * state.v + input_tensor)
    ) / (1.0 + gd2)

    # spike generation
    z_new = threshold(v_new - p.v_th, p.method, p.alpha)

    # reset
    v_out = (1 - z_new.detach()) * v_new + z_new.detach() * p.v_reset

    return z_new, WaveLIFFeedForwardState(v=v_out, v_prev=state.v)
