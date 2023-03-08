from typing import NamedTuple, Tuple

import torch

from norse.torch.functional.threshold import threshold


class CobaLIFState(NamedTuple):
    """State of a conductance based LIF neuron.

    Parameters:
        z (torch.Tensor): recurrent spikes
        v (torch.Tensor): membrane potential
        g_e (torch.Tensor): excitatory input conductance
        g_i (torch.Tensor): inhibitory input conductance
    """

    z: torch.Tensor
    v: torch.Tensor
    g_e: torch.Tensor
    g_i: torch.Tensor


default_bio_state = CobaLIFState(z=0.0, v=-65.0, g_e=0.0, g_i=0.0)


class CobaLIFParameters(NamedTuple):
    """Parameters of conductance based LIF neuron.

    Parameters:
        tau_syn_exc_inv (torch.Tensor): inverse excitatory synaptic input
                                        time constant
        tau_syn_inh_inv (torch.Tensor): inverse inhibitory synaptic input
                                        time constant
        c_m_inv (torch.Tensor): inverse membrane capacitance
        g_l (torch.Tensor): leak conductance
        e_rev_I (torch.Tensor): inhibitory reversal potential
        e_rev_E (torch.Tensor): excitatory reversal potential
        v_rest (torch.Tensor): rest membrane potential
        v_reset (torch.Tensor): reset membrane potential
        v_thresh (torch.Tensor): threshold membrane potential
        method (str): method to determine the spike threshold
                      (relevant for surrogate gradients)
        alpha (float): hyper parameter to use in surrogate gradient computation
    """

    tau_syn_exc_inv: torch.Tensor = torch.as_tensor(1.0 / 5)
    tau_syn_inh_inv: torch.Tensor = torch.as_tensor(1.0 / 5)
    c_m_inv: torch.Tensor = torch.as_tensor(1 / 0.2)
    g_l: torch.Tensor = torch.as_tensor(1 / 20 * 1 / 0.2)
    e_rev_I: torch.Tensor = torch.as_tensor(-100)
    e_rev_E: torch.Tensor = torch.as_tensor(60)
    v_rest: torch.Tensor = torch.as_tensor(-20)
    v_reset: torch.Tensor = torch.as_tensor(-70)
    v_thresh: torch.Tensor = torch.as_tensor(-10)
    method: str = "super"
    alpha: float = 100.0


default_bio_parameters = CobaLIFParameters(
    tau_syn_exc_inv=1 / 0.3,
    tau_syn_inh_inv=1 / 0.5,
    e_rev_E=0.0,
    e_rev_I=-70.0,
    v_thresh=-50.0,
    v_reset=-65.0,
    v_rest=-65.0,
)


def coba_lif_step(
    input_spikes: torch.Tensor,
    state: CobaLIFState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: CobaLIFParameters = CobaLIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, CobaLIFState]:
    """Euler integration step for a conductance based LIF neuron.

    Parameters:
        input_spikes (torch.Tensor): the input spikes at the current time step
        s (CobaLIFState): current state of the neuron
        input_weights (torch.Tensor): input weights
            (sign determines  contribution to inhibitory / excitatory input)
        recurrent_weights (torch.Tensor): recurrent weights
            (sign determines contribution to inhibitory / excitatory input)
        p (CobaLIFParameters): parameters of the neuron
        dt (float): Integration time step
    """
    # conductance jumps
    g_e = state.g_e + torch.nn.functional.linear(
        input_spikes, torch.nn.functional.relu(input_weights)
    )
    g_i = state.g_i + torch.nn.functional.linear(
        input_spikes, torch.nn.functional.relu(-input_weights)
    )

    g_e = state.g_e + torch.nn.functional.linear(
        state.z, torch.nn.functional.relu(recurrent_weights)
    )
    g_i = state.g_i + torch.nn.functional.linear(
        state.z, torch.nn.functional.relu(-recurrent_weights)
    )
    dg_e = -dt * p.tau_syn_exc_inv * g_e
    g_e = g_e + dg_e
    dg_i = -dt * p.tau_syn_inh_inv * g_i
    g_i = g_i + dg_i

    dv = (
        dt
        * p.c_m_inv
        * (
            p.g_l * (p.v_rest - state.v)
            + g_e * (p.e_rev_E - state.v)
            + g_i * (p.e_rev_I - state.v)
        )
    )
    v = state.v + dv

    z_new = threshold(v - p.v_thresh, p.method, p.alpha)
    v = (1 - z_new) * v + z_new * p.v_reset
    return z_new, CobaLIFState(z_new, v, g_e, g_i)


class CobaLIFFeedForwardState(NamedTuple):
    """State of a conductance based feed forward LIF neuron.

    Parameters:
        v (torch.Tensor): membrane potential
        g_e (torch.Tensor): excitatory input conductance
        g_i (torch.Tensor): inhibitory input conductance
    """

    v: torch.Tensor
    g_e: torch.Tensor
    g_i: torch.Tensor


def coba_lif_feed_forward_step(
    input_tensor: torch.Tensor,
    state: CobaLIFFeedForwardState,
    p: CobaLIFParameters = CobaLIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, CobaLIFFeedForwardState]:
    """Euler integration step for a conductance based LIF neuron.

    Parameters:
        input_tensor (torch.Tensor): synaptic input
        state (CobaLIFFeedForwardState): current state of the neuron
        p (CobaLIFParameters): parameters of the neuron
        dt (float): Integration time step
    """
    # conductance jumps
    g_e = state.g_e + torch.nn.functional.relu(input_tensor)
    g_i = state.g_i + torch.nn.functional.relu(-input_tensor)

    dg_e = -dt * p.tau_syn_exc_inv * g_e
    g_e = g_e + dg_e
    dg_i = -dt * p.tau_syn_inh_inv * g_i
    g_i = g_i + dg_i

    dv = (
        dt
        * p.c_m_inv
        * (
            p.g_l * (p.v_rest - state.v)
            + g_e * (p.e_rev_E - state.v)
            + g_i * (p.e_rev_I - state.v)
        )
    )
    v = state.v + dv

    z_new = threshold(v - p.v_thresh, p.method, p.alpha)
    v = (1 - z_new) * v + z_new * p.v_reset
    return z_new, CobaLIFFeedForwardState(v, g_e, g_i)
