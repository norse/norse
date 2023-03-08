from typing import NamedTuple, Tuple

import torch
import torch.jit

from norse.torch.functional.threshold import threshold


class LIFAdExParameters(NamedTuple):
    """Parametrization of an Adaptive Exponential Leaky Integrate and Fire neuron

    Default values from https://github.com/NeuralEnsemble/PyNN/blob/d8056fa956998b031a1c3689a528473ed2bc0265/pyNN/standardmodels/cells.py#L416

    Parameters:
        adaptation_current (torch.Tensor): adaptation coupling parameter in nS
        adaptation_spike (torch.Tensor): spike triggered adaptation parameter in nA
        delta_T (torch.Tensor): sharpness or speed of the exponential growth in mV
        tau_syn_inv (torch.Tensor): inverse adaptation time
                                    constant (:math:`1/\\tau_\\text{ada}`) in 1/ms
        tau_syn_inv (torch.Tensor): inverse synaptic time
                                    constant (:math:`1/\\tau_\\text{syn}`) in 1/ms
        tau_mem_inv (torch.Tensor): inverse membrane time
                                    constant (:math:`1/\\tau_\\text{mem}`) in 1/ms
        v_leak (torch.Tensor): leak potential in mV
        v_th (torch.Tensor): threshold potential in mV
        v_reset (torch.Tensor): reset potential in mV
        method (str): method to determine the spike threshold
                      (relevant for surrogate gradients)
        alpha (float): hyper parameter to use in surrogate gradient computation
    """

    adaptation_current: torch.Tensor = torch.as_tensor(4)
    adaptation_spike: torch.Tensor = torch.as_tensor(0.02)
    delta_T: torch.Tensor = torch.as_tensor(0.5)
    tau_ada_inv: torch.Tensor = torch.as_tensor(2.0)
    tau_syn_inv: torch.Tensor = torch.as_tensor(1.0 / 5e-3)
    tau_mem_inv: torch.Tensor = torch.as_tensor(1.0 / 1e-2)
    v_leak: torch.Tensor = torch.as_tensor(0.0)
    v_th: torch.Tensor = torch.as_tensor(1.0)
    v_reset: torch.Tensor = torch.as_tensor(0.0)
    method: str = "super"
    alpha: float = 100.0


class LIFAdExState(NamedTuple):
    """State of a LIFAdEx neuron

    Parameters:
        z (torch.Tensor): recurrent spikes
        v (torch.Tensor): membrane potential
        i (torch.Tensor): synaptic input current
        a (torch.Tensor): membrane potential adaptation factor
    """

    z: torch.Tensor
    v: torch.Tensor
    i: torch.Tensor
    a: torch.Tensor


class LIFAdExFeedForwardState(NamedTuple):
    """State of a feed forward LIFAdEx neuron

    Parameters:
        v (torch.Tensor): membrane potential
        i (torch.Tensor): synaptic input current
        a (torch.Tensor): membrane potential adaptation factor
    """

    v: torch.Tensor
    i: torch.Tensor
    a: torch.Tensor


def lif_adex_step(
    input_spikes: torch.Tensor,
    state: LIFAdExState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: LIFAdExParameters = LIFAdExParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFAdExState]:
    r"""Computes a single euler-integration step of an adaptive exponential LIF neuron-model
    adapted from http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model.
    More specifically it implements one integration step of the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} \left(v_{\text{leak}} - v + i + \Delta_T exp\left({{v - v_{\text{th}}} \over {\Delta_T}}\right) - a\right) \\
            \dot{i} &= -1/\tau_{\text{syn}} i \\
            \dot{a} &= 1/\tau_{\text{ada}} \left( a_{current} (V - v_{\text{leak}}) - a \right)
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}})

    and transition equations

    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\text{reset}} \\
            i &= i + w_{\text{input}} z_{\text{in}} \\
            i &= i + w_{\text{rec}} z_{\text{rec}} \\
            a &= a + a_{\text{spike}} z_{\text{rec}}
        \end{align*}

    where :math:`z_{\text{rec}}` and :math:`z_{\text{in}}` are the recurrent
    and input spikes respectively.

    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        s (LIFAdExState): current state of the LIF neuron
        input_weights (torch.Tensor): synaptic weights for incoming spikes
        recurrent_weights (torch.Tensor): synaptic weights for recurrent spikes
        p (LIFAdExParameters): parameters of a leaky integrate and fire neuron
        dt (float): Integration timestep to use
    """
    # compute current jumps
    i_jump = (
        state.i
        + torch.nn.functional.linear(input_spikes, input_weights)
        + torch.nn.functional.linear(state.z, recurrent_weights)
    )

    # compute voltage updates
    dv_leak = p.v_leak - state.v
    dv_exp = p.delta_T * torch.exp((state.v - p.v_th) / p.delta_T)
    dv = dt * p.tau_mem_inv * (dv_leak + dv_exp + i_jump - state.a)
    v_decayed = state.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * i_jump
    i_decayed = i_jump + di

    # Compute adaptation update
    da = dt * p.tau_ada_inv * (p.adaptation_current * (state.v - p.v_leak) - state.a)
    a_decayed = state.a + da

    # compute new spikes
    z_new = threshold(v_decayed - p.v_th, p.method, p.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset

    # Compute spike adaptation
    a_new = a_decayed + z_new * p.adaptation_spike

    return z_new, LIFAdExState(z_new, v_new, i_decayed, a_new)


def lif_adex_feed_forward_step(
    input_spikes: torch.Tensor,
    state: LIFAdExFeedForwardState = LIFAdExFeedForwardState(0, 0, 0),
    p: LIFAdExParameters = LIFAdExParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFAdExFeedForwardState]:
    r"""Computes a single euler-integration step of an adaptive exponential
    LIF neuron-model adapted from
    http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model.
    It takes as input the input current as generated by an arbitrary torch
    module or function. More specifically it implements one integration step of
    the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} \left(v_{\text{leak}} - v + i + \Delta_T exp\left({{v - v_{\text{th}}} \over {\Delta_T}}\right ) - a\right) \\
            \dot{i} &= -1/\tau_{\text{syn}} i \\
            \dot{a} &= 1/\tau_{\text{ada}} \left( a_{current} (V - v_{\text{leak}}) - a \right)
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}})

    and transition equations

    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\text{reset}} \\
            i &= i + i_{\text{in}} \\
            a &= a + a_{\text{spike}} z_{\text{rec}}
        \end{align*}

    where :math:`i_{\text{in}}` is meant to be the result of applying an
    arbitrary pytorch module (such as a convolution) to input spikes.

    Parameters:
        input_spikes (torch.Tensor): the input spikes at the current time step
        state (LIFAdExFeedForwardState): current state of the LIF neuron
        p (LIFAdExParameters): parameters of a leaky integrate and fire neuron
        dt (float): Integration timestep to use
    """
    # compute current jumps
    i_jump = state.i + input_spikes
    # compute voltage updates
    dv_leak = p.v_leak - state.v
    dv_exp = p.delta_T * torch.exp((state.v - p.v_th) / p.delta_T)
    dv = dt * p.tau_mem_inv * (dv_leak + dv_exp + i_jump - state.a)
    v_decayed = state.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * i_jump
    i_decayed = i_jump + di

    # Compute adaptation update
    da = dt * p.tau_ada_inv * (p.adaptation_current * (state.v - p.v_leak) - state.a)
    a_decayed = state.a + da

    # compute new spikes
    z_new = threshold(v_decayed - p.v_th, p.method, p.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset

    # compute adaptation update
    a_new = a_decayed + z_new * p.adaptation_spike

    return z_new, LIFAdExFeedForwardState(v_new, i_decayed, a_new)


def lif_adex_current_encoder(
    input_current: torch.Tensor,
    voltage: torch.Tensor,
    adaptation: torch.Tensor,
    p: LIFAdExParameters = LIFAdExParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Computes a single euler-integration step of an adaptive exponential LIF neuron-model
    adapted from http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model.
    More specifically it implements one integration step of the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} \left(v_{\text{leak}} - v + i + \Delta_T exp\left({{v - v_{\text{th}}} \over {\Delta_T}}\right) - a\right) \\
            \dot{i} &= -1/\tau_{\text{syn}} i \\
            \dot{a} &= 1/\tau_{\text{ada}} \left( a_{current} (V - v_{\text{leak}}) - a \right)
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}})

    and transition equations

    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\text{reset}} \\
            i &= i + i_{\text{in}} \\
            a &= a + a_{\text{spike}} z_{\text{rec}}
        \end{align*}


    Parameters:
        input_current (torch.Tensor): the input current at the current time step
        voltage (torch.Tensor): current state of the LIFAdEx neuron
        adaptation (torch.Tensor): membrane adaptation parameter in nS
        p (LIFAdExParameters): parameters of a leaky integrate and fire neuron
        dt (float): Integration timestep to use
    """
    dv_leak = p.v_leak - voltage
    dv_exp = p.delta_T * torch.exp((voltage - p.v_th) / p.delta_T)
    dv = dt * p.tau_mem_inv * (dv_leak + dv_exp + input_current - adaptation)
    voltage = voltage + dv
    z = threshold(voltage - p.v_th, p.method, p.alpha)

    voltage = voltage - z * (voltage - p.v_reset)
    adaptation = (
        p.tau_ada_inv * (p.adaptation_current * (voltage - p.v_leak) - adaptation)
        + z * p.adaptation_spike
    )
    return z, voltage, adaptation
