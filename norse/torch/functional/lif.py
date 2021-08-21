r"""
A very popular neuron model that combines a :mod:`norse.torch.functional.leaky_integrator` with
spike thresholds to produce events (spikes).

The model describes the change in a neuron membrane voltage (:math:`v`)
and inflow current (:math:`i`).
See the :mod:`.leaky_integrator` module for more information.

.. math::
    \begin{align*}
        \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
        \dot{i} &= 1/\tau_{\text{syn}} i
    \end{align*}

The F in LIF stands for the thresholded "firing" events that occur if the
neuron voltage increases over a certain point or *threshold* (:math:`v_{\text{th}}`).

.. math::
    z = \Theta(v - v_{\text{th}})

In regular artificial neural networks, this is referred to as the *activation
function*. The behaviour can be controlled by setting the :code:`method` field in
the neuron parameters, but will default to the :mod:`.superspike` synthetic
gradient approach that uses the :mod:`.heaviside` step function:

.. math::
    H[n]=\begin{cases} 0, & n <= 0 \\ 1, & n \gt 0 \end{cases}

"""
from typing import NamedTuple, Optional, Tuple
import torch
import torch.jit

try:
    import norse_op
except ModuleNotFoundError:  # pragma: no cover
    pass

from norse.torch.functional.threshold import threshold
from norse.torch.functional.lift import lift
import norse.utils


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
    alpha: float = torch.as_tensor(100.0)


default_bio_parameters = LIFParameters(
    tau_syn_inv=torch.as_tensor(1 / 0.5),
    tau_mem_inv=torch.as_tensor(1 / 20.0),
    v_leak=torch.as_tensor(-65.0),
    v_th=torch.as_tensor(-50.0),
    v_reset=torch.as_tensor(-65.0),
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


default_bio_initial_state = LIFState(
    z=torch.as_tensor(0.0), v=torch.as_tensor(-65.0), i=torch.as_tensor(0.0)
)


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
    v_new = (1 - z_new.detach()) * v_decayed + z_new.detach() * p.v_reset
    # compute current jumps
    i_new = (
        i_decayed
        + torch.nn.functional.linear(input_tensor, input_weights)
        + torch.nn.functional.linear(state.z, recurrent_weights)
    )

    return z_new, LIFState(z_new, v_new, i_new)


def lif_step_sparse(
    input_tensor: torch.Tensor,
    state: LIFState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: LIFParameters = LIFParameters(),
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
        + torch.sparse.mm(input_tensor, input_weights.t())
        + torch.sparse.mm(state.z, recurrent_weights.t())
    )

    z_sparse = z_new.to_sparse()
    return z_sparse, LIFState(z_sparse, v_new, i_new)


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

    if norse.utils.IS_OPS_LOADED:
        try:
            z, v, i = norse_op.lif_super_step(
                input_tensor, state, input_weights, recurrent_weights, p, dt
            )
            return z, LIFState(z=z, v=v, i=i)
        except NameError:  # pragma: no cover
            pass
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


def lif_step_integral(
    input_tensor: torch.Tensor,
    state: LIFState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: LIFParameters = LIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFState]:
    r"""Computes multiple euler-integration steps of a LIF neuron-model. More
    specifically it integrates the following ODE

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
        input_tensor (torch.Tensor): the input spikes, assuming the outer (first) dimension is time
        s (LIFState): current state of the LIF neuron
        input_weights (torch.Tensor): synaptic weights for incoming spikes
        recurrent_weights (torch.Tensor): synaptic weights for recurrent spikes
        p (LIFParameters): parameters of a leaky integrate and fire neuron
        dt (float): Integration timestep to use
    
    Returns:
        A tuple of (spike output from all timesteps, neuron state from the final timestep)
    """
    if norse.utils.IS_OPS_LOADED:
        try:
            z, v, i = norse_op.lif_super_integral(
                input_tensor, state, input_weights, recurrent_weights, p, dt
            )
            return z, LIFState(z=z, v=v, i=i)
        except NameError:  # pragma: no cover
            pass
    return lift(_lif_step_jit)(
        input_tensor=input_tensor,
        state=state,
        input_weights=input_weights,
        recurrent_weights=recurrent_weights,
        p=p,
        dt=dt,
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
    state: Optional[LIFFeedForwardState],
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
    if norse.utils.IS_OPS_LOADED:
        z, v, i = norse_op.lif_super_feed_forward_step(input_tensor, state, p, dt)
        return z, LIFFeedForwardState(v=v, i=i)
    jit_params = LIFParametersJIT(
        tau_syn_inv=p.tau_syn_inv,
        tau_mem_inv=p.tau_mem_inv,
        v_leak=p.v_leak,
        v_th=p.v_th,
        v_reset=p.v_reset,
        method=p.method,
        alpha=torch.as_tensor(p.alpha),
    )
    return _lif_feed_forward_step_jit(input_tensor, state=state, p=jit_params, dt=dt)


def lif_feed_forward_integral(
    input_tensor: torch.Tensor,
    state: LIFFeedForwardState,
    p: LIFParameters = LIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFState]:
    r"""Computes multiple euler-integration steps of a LIF neuron-model. More
    specifically it integrates the following ODE

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

    Parameters:
        input_tensor (torch.Tensor): the input spikes with the outer dimension assumed to be timesteps
        s (LIFState): current state of the LIF neuron
        p (LIFParameters): parameters of a leaky integrate and fire neuron
        dt (float): Integration timestep to use
    """
    if norse.utils.IS_OPS_LOADED:
        try:
            z, v, i = norse_op.lif_super_feed_forward_integral(
                input_tensor, state, p, dt
            )
            return z, LIFState(z=z, v=v, i=i)
        except NameError:  # pragma: no cover
            pass
    return lift(lif_feed_forward_step)(
        input_tensor=input_tensor, state=state, p=p, dt=dt
    )


def lif_feed_forward_step_sparse(
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
    thresholds = (v_decayed - p.v_th).coalesce()
    jumps = threshold(thresholds.values(), p.method, p.alpha)
    z_new = torch.sparse_coo_tensor(
        indices=thresholds.indices(),
        values=jumps,
        size=v_decayed.size(),
        device=thresholds.device,
    ).coalesce()
    # z_new = threshold((v_decayed - p.v_th).to_dense(), p.method, p.alpha)
    # z_new = z_new.to_sparse().coalesce()
    # compute reset
    ones = torch.sparse_coo_tensor(
        indices=z_new.indices(),
        values=torch.full_like(z_new.values(), 1),
        size=z_new.size(),
        device=z_new.device,
    )
    v_new = (ones - z_new) * v_decayed + z_new * p.v_reset
    # compute current jumps
    i_new = i_decayed + input_tensor

    return z_new, LIFFeedForwardState(v=v_new, i=i_new)


# # compute new spikes
#     thresholds = (v_decayed - p.v_th).coalesce()
#     jumps = threshold(thresholds.values(), p.method, p.alpha)
#     z_new = torch.sparse_coo_tensor(
#         indices=thresholds.indices(),
#         values=jumps,
#         size=v_decayed.size(),
#         device=thresholds.device,
#     ).coalesce()

#     # compute reset
#     ones = torch.sparse_coo_tensor(
#             indices=z_new.indices(),
#             values=torch.full_like(z_new.values(), 1),
#             size=z_new.size(),
#             device=z_new.device,
#         )
#     v_new = (ones - z_new) * v_decayed + z_new * p.v_reset
#     # compute current jumps
#     i_new = i_decayed + input_tensor

#     return z_new.to_sparse().coalesce(), LIFFeedForwardState(v=v_new, i=i_new)


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
