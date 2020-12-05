from typing import NamedTuple, Optional, Tuple

import torch
import torch.jit
import norse
from norse.torch.functional.threshold import threshold


class LIFParameters(NamedTuple):
    """Parametrization of a LIF neuron

    Parameters:
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

    tau_syn_inv: torch.Tensor = torch.as_tensor(1.0 / 5e-3)
    tau_mem_inv: torch.Tensor = torch.as_tensor(1.0 / 1e-2)
    v_leak: torch.Tensor = torch.as_tensor(0.0)
    v_th: torch.Tensor = torch.as_tensor(1.0)
    v_reset: torch.Tensor = torch.as_tensor(0.0)
    method: str = "super"
    alpha: float = torch.as_tensor(0.0)


default_bio_parameters = LIFParameters(
    tau_syn_inv=1 / 0.5,
    tau_mem_inv=1 / 20.0,
    v_leak=-65.0,
    v_th=-50.0,
    v_reset=-65.0,
)


class LIFState(NamedTuple):
    """State of a LIF neuron

    Parameters:
        z (torch.Tensor): recurrent spikes
        v (torch.Tensor): membrane potential
        i (torch.Tensor): synaptic input current
    """

    z: torch.Tensor
    v: torch.Tensor
    i: torch.Tensor


default_bio_initial_state = LIFState(z=0.0, v=-65.0, i=0.0)


class LIFFeedForwardState(NamedTuple):
    """State of a feed forward LIF neuron

    Parameters:
        v (torch.Tensor): membrane potential
        i (torch.Tensor): synaptic input current
    """

    v: torch.Tensor
    i: torch.Tensor


class LIFParametersJIT(NamedTuple):
    """Parametrization of a LIF neuron

    Parameters:
        tau_syn_inv (torch.Tensor): inverse synaptic time
                                    constant (:math:`1/\\tau_\\text{syn}`) in 1/ms
        tau_mem_inv (torch.Tensor): inverse membrane time
                                    constant (:math:`1/\\tau_\\text{mem}`) in 1/ms
        v_leak (torch.Tensor): leak potential in mV
        v_th (torch.Tensor): threshold potential in mV
        v_reset (torch.Tensor): reset potential in mV
        method (str): method to determine the spike threshold
                      (relevant for surrogate gradients)
        alpha (torch.Tensor): hyper parameter to use in surrogate gradient computation
    """

    tau_syn_inv: torch.Tensor
    tau_mem_inv: torch.Tensor
    v_leak: torch.Tensor
    v_th: torch.Tensor
    v_reset: torch.Tensor
    method: str
    alpha: torch.Tensor


@torch.jit.script
def _lif_step_jit(
    input_tensor: torch.Tensor,
    state: LIFState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: LIFParametersJIT,
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFState]:  # pragma: no cover
    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + state.i)
    v_decayed = state.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * state.i
    i_decayed = state.i + di

    # compute new spikes
    z_new = threshold(v_decayed - p.v_th, p.method, p.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset
    # compute current jumps
    i_new = (
        i_decayed
        + torch.nn.functional.linear(input_tensor, input_weights)
        + torch.nn.functional.linear(state.z, recurrent_weights)
    )

    return z_new, LIFState(z_new, v_new, i_new)


def lif_step(
    input_tensor: torch.Tensor,
    state: LIFState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: LIFParameters = LIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFState]:
    r"""Computes a single euler-integration step of a LIF neuron-model. More
    specifically it implements one integration step of the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}})

    and transition equations

    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\text{reset}} \\
            i &= i + w_{\text{input}} z_{\text{in}} \\
            i &= i + w_{\text{rec}} z_{\text{rec}}
        \end{align*}

    where :math:`z_{\text{rec}}` and :math:`z_{\text{in}}` are the recurrent
    and input spikes respectively.

    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        s (LIFState): current state of the LIF neuron
        input_weights (torch.Tensor): synaptic weights for incoming spikes
        recurrent_weights (torch.Tensor): synaptic weights for recurrent spikes
        p (LIFParameters): parameters of a leaky integrate and fire neuron
        dt (float): Integration timestep to use
    """
    if p.method == "super" and getattr(norse, "IS_OPS_LOADED"):
        # Order: [v_leak, v_th, v_reset, tau_mem_inv, tau_syn_inv, alpha]
        cpp_params = (
            p.v_leak,
            p.v_th,
            p.v_reset,
            p.tau_mem_inv,
            p.tau_syn_inv,
            torch.as_tensor(p.alpha),
        )
        z, v, i = torch.ops.norse_op.lif_super_step(
            input_tensor, state, input_weights, recurrent_weights, cpp_params, dt
        )
        return z, LIFState(z=z, v=v, i=i)
    else:
        jit_params = LIFParametersJIT(
            tau_syn_inv=p.tau_syn_inv,
            tau_mem_inv=p.tau_mem_inv,
            v_leak=p.v_leak,
            v_th=p.v_th,
            v_reset=p.v_reset,
            method=p.method,
            alpha=torch.as_tensor(p.alpha),
        )
        return _lif_step_jit(
            input_tensor, state, input_weights, recurrent_weights, jit_params, dt
        )


@torch.jit.script
def _lif_feed_forward_step_jit(
    input_tensor: torch.Tensor,
    state: LIFFeedForwardState,
    p: LIFParametersJIT,
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFFeedForwardState]:  # pragma: no cover
    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + state.i)
    v_decayed = state.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * state.i
    i_decayed = state.i + di

    # compute new spikes
    z_new = threshold(v_decayed - p.v_th, p.method, p.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset
    # compute current jumps
    i_new = i_decayed + input_tensor

    return z_new, LIFFeedForwardState(v=v_new, i=i_new)


def lif_feed_forward_step(
    input_tensor: torch.Tensor,
    state: Optional[LIFFeedForwardState] = None,
    p: LIFParameters = LIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFFeedForwardState]:
    r"""Computes a single euler-integration step for a lif neuron-model.
    It takes as input the input current as generated by an arbitrary torch
    module or function. More specifically it implements one integration
    step of the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}})

    and transition equations

    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\text{reset}} \\
            i &= i + i_{\text{in}}
        \end{align*}

    where :math:`i_{\text{in}}` is meant to be the result of applying an
    arbitrary pytorch module (such as a convolution) to input spikes.

    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        state (LIFFeedForwardState): current state of the LIF neuron
        p (LIFParameters): parameters of a leaky integrate and fire neuron
        dt (float): Integration timestep to use
    """
    jit_params = LIFParametersJIT(
        tau_syn_inv=p.tau_syn_inv,
        tau_mem_inv=p.tau_mem_inv,
        v_leak=p.v_leak,
        v_th=p.v_th,
        v_reset=p.v_reset,
        method=p.method,
        alpha=torch.as_tensor(p.alpha),
    )
    # Because input tensors are not directly used in the first pass (no
    # broadcasting takes place) we need to set the state values to the
    # same shape as the input.
    if state is None:
        state = LIFFeedForwardState(
            v=torch.full_like(input_tensor, jit_params.v_reset),
            i=torch.zeros_like(input_tensor),
        )
    return _lif_feed_forward_step_jit(input_tensor, state=state, p=jit_params, dt=dt)


def lif_current_encoder(
    input_current: torch.Tensor,
    voltage: torch.Tensor,
    p: LIFParameters = LIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Computes a single euler-integration step of a leaky integrator. More
    specifically it implements one integration step of the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}

    Parameters:
        input (torch.Tensor): the input current at the current time step
        voltage (torch.Tensor): current state of the LIF neuron
        p (LIFParameters): parameters of a leaky integrate and fire neuron
        dt (float): Integration timestep to use
    """
    dv = dt * p.tau_mem_inv * ((p.v_leak - voltage) + input_current)
    voltage = voltage + dv
    z = threshold(voltage - p.v_th, p.method, p.alpha)

    voltage = voltage - z * (voltage - p.v_reset)
    return z, voltage
