r"""
A popular neuron model that combines a :mod:`norse.torch.functional.leaky_integrator` with
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


More information can be found on
`Wikipedia <https://en.wikipedia.org/wiki/Biological_neuron_model#Leaky_integrate-and-fire>`_
or in the book `*Neuron Dynamics* by W. Gerstner et al.,
freely available online <https://neuronaldynamics.epfl.ch/online/Ch5.html>`_.
"""

from typing import NamedTuple, Tuple
import torch
import torch.jit

from norse.torch.functional.threshold import threshold
import norse.torch.utils.pytree as pytree


class LIFParameters(
    pytree.StateTuple, metaclass=pytree.MultipleInheritanceNamedTupleMeta
):
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


# pytype: disable=bad-unpacking,wrong-keyword-args
default_bio_parameters = LIFParameters(
    tau_syn_inv=torch.as_tensor(1 / 0.5),
    tau_mem_inv=torch.as_tensor(1 / 20.0),
    v_leak=torch.as_tensor(-65.0),
    v_th=torch.as_tensor(-50.0),
    v_reset=torch.as_tensor(-65.0),
)
# pytype: enable=bad-unpacking,wrong-keyword-args


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


def lif_step_sparse(
    input_spikes: torch.Tensor,
    state: LIFState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: LIFParameters,
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
        input_spikes (torch.SparseTensor): the input spikes at the current time step
        s (LIFState): current state of the LIF neuron
        input_weights (torch.Tensor): synaptic weights for incoming spikes
        recurrent_weights (torch.Tensor): synaptic weights for recurrent spikes
        p (LIFParameters): parameters of a leaky integrate and fire neuron
        dt (float): Integration timestep to use
    """
    # compute current jumps
    i_jump = (
        state.i
        + torch.sparse.mm(input_spikes, input_weights.t())
        + torch.sparse.mm(state.z, recurrent_weights.t())
    )

    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + i_jump)
    v_decayed = state.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * i_jump
    i_decayed = i_jump + di

    # compute new spikes
    z_new = threshold(v_decayed - p.v_th, p.method, p.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset

    z_sparse = z_new.to_sparse()
    return z_sparse, LIFState(z_sparse, v_new, i_decayed)


def lif_step(
    input_spikes: torch.Tensor,
    state: LIFState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: LIFParameters,
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
        input_spikes (torch.Tensor): the input spikes at the current time step
        s (LIFState): current state of the LIF neuron
        input_weights (torch.Tensor): synaptic weights for incoming spikes
        recurrent_weights (torch.Tensor): synaptic weights for recurrent spikes
        p (LIFParameters): parameters of a leaky integrate and fire neuron
        dt (float): Integration timestep to use
    """
    # compute current jumps
    i_jump = (
        state.i
        + torch.nn.functional.linear(input_spikes, input_weights)
        + torch.nn.functional.linear(state.z, recurrent_weights)
    )
    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + i_jump)
    v_decayed = state.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * i_jump
    i_decayed = i_jump + di

    # compute new spikes
    z_new = threshold(v_decayed - p.v_th, p.method, p.alpha)
    # compute reset
    v_new = (1 - z_new.detach()) * v_decayed + z_new.detach() * p.v_reset

    return z_new, LIFState(z_new, v_new, i_decayed)


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
    out = []
    for t in input_tensor:
        z, state = lif_step(t, state, input_weights, recurrent_weights, p, dt=dt)
        out.append(z)
    return z, state


def lif_feed_forward_step(
    input_spikes: torch.Tensor,
    state: LIFFeedForwardState,
    p: LIFParameters,
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
    # compute current jumps
    i_new = state.i + input_spikes
    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + i_new)
    v_decayed = state.v + dv
    # compute current updates
    di = -dt * p.tau_syn_inv * i_new
    i_decayed = i_new + di
    # compute new spikes
    z_new = threshold(v_decayed - p.v_th, p.method, p.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset

    return z_new, LIFFeedForwardState(v=v_new, i=i_decayed)


def lif_feed_forward_integral(
    input_tensor: torch.Tensor,
    state: LIFFeedForwardState,
    p: LIFParameters,
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFFeedForwardState]:
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
    outputs = []

    for input_spikes in input_tensor:
        # compute current jumps
        i_new = state.i + input_spikes
        # compute voltage updates
        dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + i_new)
        v_decayed = state.v + dv

        # compute current updates
        di = -dt * p.tau_syn_inv * i_new
        i_decayed = i_new + di

        # compute new spikes
        z_new = threshold(v_decayed - p.v_th, p.method, p.alpha)
        # compute reset
        v_new = (1 - z_new) * v_decayed + z_new * p.v_reset

        outputs.append(z_new)
        state = LIFFeedForwardState(v=v_new, i=i_decayed)
    return torch.stack(outputs), state


def lif_feed_forward_step_sparse(
    input_tensor: torch.Tensor,
    state: LIFFeedForwardState,
    p: LIFParameters,
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFFeedForwardState]:  # pragma: no cover
    # compute current jumps
    i_jump = state.i + input_tensor

    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + i_jump)
    v_decayed = state.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * i_jump
    i_decayed = i_jump + di

    # compute new spikes
    z_new = threshold(v_decayed - p.v_th, p.method, p.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset

    return z_new.to_sparse(), LIFFeedForwardState(v=v_new, i=i_decayed)


def lif_current_encoder(
    input_current: torch.Tensor,
    voltage: torch.Tensor,
    p: LIFParameters,
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
