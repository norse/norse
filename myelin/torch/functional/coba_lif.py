import torch

from .threshhold import threshhold

from typing import NamedTuple, Tuple


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


class CobaLIFParameters(NamedTuple):
    """Parameters of conductance based LIF neuron.

    Parameters:
        tau_syn_exc_inv (torch.Tensor): inverse excitatory synaptic input time constant
        tau_syn_inh_inv (torch.Tensor): inverse inhibitory synaptic input time constant
        c_m_inv (torch.Tensor): inverse membrane capacitance
        g_l (torch.Tensor): leak conductance
        e_rev_I (torch.Tensor): inhibitory reversal potential
        e_rev_E (torch.Tensor): excitatory reversal potential
        v_rest (torch.Tensor): rest membrane potential
        v_reset (torch.Tensor): reset membrane potential
        v_thresh (torch.Tensor): threshhold membrane potential
        method (str): method to determine the spike threshold (relevant for surrogate gradients)
        alpha (float): hyper parameter to use in surrogate gradient computation
    """

    tau_syn_exc_inv: torch.Tensor = torch.tensor(1.0 / 5)
    tau_syn_inh_inv: torch.Tensor = torch.tensor(1.0 / 5)
    c_m_inv: torch.Tensor = torch.tensor(1 / 0.2)
    g_l: torch.Tensor = torch.tensor(1 / 20 * 1 / 0.2)
    e_rev_I: torch.Tensor = torch.tensor(-100)
    e_rev_E: torch.Tensor = torch.tensor(60)
    v_rest: torch.Tensor = torch.tensor(-20)
    v_reset: torch.Tensor = torch.tensor(-70)
    v_thresh: torch.Tensor = torch.tensor(-10)
    method: str = "heaviside"
    alpha: float = 0.0


def coba_lif_step(
    input: torch.Tensor,
    s: CobaLIFState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: CobaLIFParameters = CobaLIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, CobaLIFState]:
    """Euler integration step for a conductance based LIF neuron.

    Parameters:
        input (torch.Tensor): the input spikes at the current time step
        s (CobaLIFState): current state of the neuron
        input_weights (torch.Tensor): input weights (sign determines contribution to inhibitory / excitatory input)
        recurrent_weights (torch.Tensor): recurrent weights (sign determines contribution to inhibitory / excitatory input)
        p (CobaLIFParameters): parameters of the neuron
        dt (float): Integration time step
    """
    dg_e = -dt * p.tau_syn_exc_inv * s.g_e
    g_e = s.g_e + dg_e
    dg_i = -dt * p.tau_syn_inh_inv * s.g_i
    g_i = s.g_i + dg_i

    g_e = g_e + torch.nn.functional.linear(
        input, torch.nn.functional.relu(input_weights)
    )
    g_i = g_i + torch.nn.functional.linear(
        input, torch.nn.functional.relu(-input_weights)
    )

    g_e = g_e + torch.nn.functional.linear(
        s.z, torch.nn.functional.relu(recurrent_weights)
    )
    g_i = g_i + torch.nn.functional.linear(
        s.z, torch.nn.functional.relu(-recurrent_weights)
    )

    dv = (
        dt
        * p.c_m_inv
        * (p.g_l * (p.v_rest - s.v) + g_e * (p.e_rev_E - s.v) + g_i * (p.e_rev_I - s.v))
    )
    v = s.v + dv

    z_new = threshhold(v - p.v_thresh, p.method, p.alpha)
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
    input: torch.Tensor,
    s: CobaLIFFeedForwardState,
    p: CobaLIFParameters = CobaLIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, CobaLIFFeedForwardState]:
    """Euler integration step for a conductance based LIF neuron.

    Parameters:
        input (torch.Tensor): synaptic input
        s (CobaLIFFeedForwardState): current state of the neuron
        p (CobaLIFParameters): parameters of the neuron
        dt (float): Integration time step
    """
    dg_e = -dt * p.tau_syn_exc_inv * s.g_e
    g_e = s.g_e + dg_e
    dg_i = -dt * p.tau_syn_inh_inv * s.g_i
    g_i = s.g_i + dg_i

    g_e = g_e + torch.nn.functional.relu(input)
    g_i = g_i + torch.nn.functional.relu(-input)

    dv = (
        dt
        * p.c_m_inv
        * (p.g_l * (p.v_rest - s.v) + g_e * (p.e_rev_E - s.v) + g_i * (p.e_rev_I - s.v))
    )
    v = s.v + dv

    z_new = threshhold(v - p.v_thresh, p.method, p.alpha)
    v = (1 - z_new) * v + z_new * p.v_reset
    return z_new, CobaLIFFeedForwardState(v, g_e, g_i)
