r"""
Long-short term memory module, building on the work by
[G. Bellec, D. Salaj, A. Subramoney, R. Legenstein, and W. Maass](https://github.com/IGITUGraz/LSNN-official).

The LSNN dynamics is similar to the :mod:`.lif` equations, but it
adds an adaptive term :math:`b`:

.. math::
    \begin{align*}
        \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
        \dot{i} &= -1/\tau_{\text{syn}}\; i \\
        \dot{b} &= -1/\tau_{b}\; b
    \end{align*}

This adaptation is applied in the jump condition when the neuron spikes:

.. math::
    z = \Theta(v - v_{\text{th}} + b)

Contrast this with the regular LIF jump condition:

.. math::
    z = \Theta(v - v_{\text{th}})

In practice, this means that the LSNN neurons *adapt* to fire more or less
given the same input. The adaptation is determined by the :math:`\tau_b`
time constant.
"""

from typing import NamedTuple, Tuple

import torch

from norse.torch.functional.threshold import threshold


class LSNNParameters(NamedTuple):
    r"""Parameters of an LSNN neuron

    Parameters:
        tau_syn_inv (torch.Tensor): inverse synaptic time
                                    constant (:math:`1/\tau_\text{syn}`)
        tau_mem_inv (torch.Tensor): inverse membrane time
                                    constant (:math:`1/\tau_\text{mem}`)
        tau_adapt_inv (torch.Tensor): adaptation time constant (:math:`\tau_b`)
        v_leak (torch.Tensor): leak potential
        v_th (torch.Tensor): threshold potential
        v_reset (torch.Tensor): reset potential
        beta (torch.Tensor): adaptation constant
    """

    tau_syn_inv: torch.Tensor = torch.as_tensor(1.0 / 5e-3)
    tau_mem_inv: torch.Tensor = torch.as_tensor(1.0 / 1e-2)
    tau_adapt_inv: torch.Tensor = torch.as_tensor(1.0 / 800)
    v_leak: torch.Tensor = torch.as_tensor(0.0)
    v_th: torch.Tensor = torch.as_tensor(1.0)
    v_reset: torch.Tensor = torch.as_tensor(0.0)
    beta: torch.Tensor = torch.as_tensor(1.8)
    method: str = "super"
    alpha: float = 100.0


class LSNNState(NamedTuple):
    """State of an LSNN neuron

    Parameters:
        z (torch.Tensor): recurrent spikes
        v (torch.Tensor): membrane potential
        i (torch.Tensor): synaptic input current
        b (torch.Tensor): threshold adaptation
    """

    z: torch.Tensor
    v: torch.Tensor
    i: torch.Tensor
    b: torch.Tensor


def lsnn_step(
    input_tensor: torch.Tensor,
    state: LSNNState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: LSNNParameters = LSNNParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LSNNState]:
    r"""Euler integration step for LIF Neuron with threshold adaptation
    More specifically it implements one integration step of the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i \\
            \dot{b} &= -1/\tau_{b} b
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}} + b)

    and transition equations

    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\text{reset}} \\
            i &= i + w_{\text{input}} z_{\text{in}} \\
            i &= i + w_{\text{rec}} z_{\text{rec}} \\
            b &= b + \beta z
        \end{align*}

    where :math:`z_{\text{rec}}` and :math:`z_{\text{in}}` are the recurrent
    and input spikes respectively.

    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        s (LSNNState): current state of the lsnn unit
        input_weights (torch.Tensor): synaptic weights for input spikes
        recurrent_weights (torch.Tensor): synaptic weights for recurrent spikes
        p (LSNNParameters): parameters of the lsnn unit
        dt (float): Integration timestep to use
    """
    # compute voltage decay
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + state.i)
    v_decayed = state.v + dv

    # compute current decay
    di = -dt * p.tau_syn_inv * state.i
    i_decayed = state.i + di

    # compute threshold adaptation update
    db = dt * p.tau_adapt_inv * (p.v_th - state.b)
    b_decayed = state.b + db

    # compute new spikes
    z_new = threshold(v_decayed - b_decayed, p.method, p.alpha)
    # compute reset
    v_new = (1 - z_new.detach()) * v_decayed + z_new.detach() * p.v_reset
    # compute current jumps
    i_new = (
        i_decayed
        + torch.nn.functional.linear(input_tensor, input_weights)
        + torch.nn.functional.linear(state.z, recurrent_weights)
    )

    b_new = b_decayed + z_new.detach() * p.beta
    return z_new, LSNNState(z_new, v_new, i_new, b_new)


def ada_lif_step(
    input_tensor: torch.Tensor,
    state: LSNNState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: LSNNParameters = LSNNParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LSNNState]:
    r"""Euler integration step for LIF Neuron with adaptation. More specifically
    it implements one integration step of the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + b + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i \\
            \dot{b} &= -1/\tau_{b} b
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}})

    and transition equations

    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\\text{reset}} \\
            i &= i + w_{\text{input}} z_{\\text{in}} \\
            i &= i + w_{\text{rec}} z_{\\text{rec}} \\
            b &= b + \beta z
        \end{align*}

    where :math:`z_{\text{rec}}` and :math:`z_{\text{in}}` are the recurrent
    and input spikes respectively.

    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        s (LSNNState): current state of the lsnn unit
        input_weights (torch.Tensor): synaptic weights for input spikes
        recurrent_weights (torch.Tensor): synaptic weights for recurrent spikes
        p (LSNNParameters): parameters of the lsnn unit
        dt (float): Integration timestep to use
    """
    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + state.i - state.b)
    v_decayed = state.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * state.i
    i_decayed = state.i + di

    # compute threshold updates
    db = -dt * p.tau_adapt_inv * state.b
    b_decayed = state.b + db

    # compute new spikes
    z_new = threshold(v_decayed - p.v_th, p.method, p.alpha)
    # compute resets
    v_new = v_decayed - z_new * (p.v_th - p.v_reset)
    # compute b update
    b_new = b_decayed + z_new * p.beta
    # compute current jumps
    i_new = (
        i_decayed
        + torch.nn.functional.linear(input_tensor, input_weights)
        + torch.nn.functional.linear(state.z, recurrent_weights)
    )
    return z_new, LSNNState(z_new, v_new, i_new, b_new)


class LSNNFeedForwardState(NamedTuple):
    """Integration state kept for a lsnn module

    Parameters:
        v (torch.Tensor): membrane potential
        i (torch.Tensor): synaptic input current
        b (torch.Tensor): threshold adaptation
    """

    v: torch.Tensor
    i: torch.Tensor
    b: torch.Tensor


def lsnn_feed_forward_step(
    input_tensor: torch.Tensor,
    state: LSNNFeedForwardState,
    p: LSNNParameters = LSNNParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LSNNFeedForwardState]:
    r"""Euler integration step for LIF Neuron with threshold adaptation.
    More specifically it implements one integration step of the following ODE

    .. math::
        \\begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i \\
            \dot{b} &= -1/\tau_{b} b
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}} + b)

    and transition equations

    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\text{reset}} \\
            i &= i + \text{input} \\
            b &= b + \beta z
        \end{align*}

    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        s (LSNNFeedForwardState): current state of the lsnn unit
        p (LSNNParameters): parameters of the lsnn unit
        dt (float): Integration timestep to use
    """
    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + state.i)
    v_decayed = state.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * state.i
    i_decayed = state.i + di

    # compute threshold updates
    db = dt * p.tau_adapt_inv * (p.v_th - state.b)
    b_decayed = state.b + db

    # compute new spikes
    z_new = threshold(v_decayed - b_decayed, p.method, p.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset
    # compute b update
    b_new = b_decayed + z_new * p.beta
    # compute current jumps
    i_new = i_decayed + input_tensor

    return z_new, LSNNFeedForwardState(v=v_new, i=i_new, b=b_new)
