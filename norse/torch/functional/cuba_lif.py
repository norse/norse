r"""
A current-based leaky integrate-and-fire neuron model.
Compared to the :mod:`norse.torch.functional.lif` module, this model exposes
an input weight :math:`w_{\text{in}}` and a resistance :math:`R` scaling the
synaptic current, and it integrates the membrane with the *updated*
synaptic current, so inputs affect the membrane in the same timestep.

.. math::
    \begin{align*}
        \tau_{\text{syn}} \dot{i} &= -i + w_{\text{in}} S \\
        \tau_{\text{mem}} \dot{v} &= (v_{\text{leak}} - v) + R i
    \end{align*}

together with the jump condition

.. math::
    z = \Theta(v - v_{\text{th}})

and transition equations

.. math::
    v = (1-z) v + z v_{\text{reset}}
"""

from typing import Tuple
import torch
import torch.jit

from norse.torch.functional.threshold import threshold
from norse.torch.functional.reset import ResetMethod, reset_value
from norse.torch.utils import pytree


class CubaLIFParameters(
    pytree.StateTuple, metaclass=pytree.MultipleInheritanceNamedTupleMeta
):
    """Parametrization of a current-based LIF neuron

    Parameters:
        tau_syn_inv (torch.Tensor): inverse synaptic time
                                    constant (:math:`1/\\tau_\\text{syn}`) in 1/ms
        tau_mem_inv (torch.Tensor): inverse membrane time
                                    constant (:math:`1/\\tau_\\text{mem}`) in 1/ms
        r (torch.Tensor): resistance scaling the synaptic current into the membrane
        w_in (torch.Tensor): input weight scaling the incoming spikes
        v_leak (torch.Tensor): leak potential in mV
        v_th (torch.Tensor): threshold potential in mV
        v_reset (torch.Tensor): reset potential in mV
        method (str): method to determine the spike threshold
                      (relevant for surrogate gradients)
        alpha (float): hyper parameter to use in surrogate gradient computation
        reset_method (ResetMethod): method to reset the membrane after a spike
    """

    tau_syn_inv: torch.Tensor = torch.as_tensor(1.0 / 5e-3)
    tau_mem_inv: torch.Tensor = torch.as_tensor(1.0 / 1e-2)
    r: torch.Tensor = torch.as_tensor(1.0)
    w_in: torch.Tensor = torch.as_tensor(1.0)
    v_leak: torch.Tensor = torch.as_tensor(0.0)
    v_th: torch.Tensor = torch.as_tensor(1.0)
    v_reset: torch.Tensor = torch.as_tensor(0.0)
    method: str = "super"
    alpha: float = torch.as_tensor(100.0)
    reset_method: ResetMethod = reset_value


class CubaLIFFeedForwardState(
    pytree.StateTuple, metaclass=pytree.MultipleInheritanceNamedTupleMeta
):
    """State of a feed forward current-based LIF neuron

    Parameters:
        v (torch.Tensor): membrane potential
        i (torch.Tensor): synaptic input current
    """

    v: torch.Tensor
    i: torch.Tensor


def cuba_lif_feed_forward_step(
    input_tensor: torch.Tensor,
    state: CubaLIFFeedForwardState,
    p: CubaLIFParameters = CubaLIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, CubaLIFFeedForwardState]:  # pragma: no cover
    r"""Computes a single euler-integration step for a current-based lif
    neuron-model. More specifically it implements one integration step of
    the following ODE

    .. math::
        \begin{align*}
            \tau_{\text{syn}} \dot{i} &= -i + w_{\text{in}} S \\
            \tau_{\text{mem}} \dot{v} &= (v_{\text{leak}} - v) + R i
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}})

    and transition equations

    .. math::
        v = (1-z) v + z v_{\text{reset}}

    The input enters the synaptic current as an instantaneous jump of
    :math:`w_{\text{in}} S` and is *not* scaled by :math:`dt / \tau_{\text{syn}}`.
    The membrane is then integrated with the updated synaptic current.

    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        state (CubaLIFFeedForwardState): current state of the neuron
        p (CubaLIFParameters): parameters of the current-based lif neuron
        dt (float): Integration timestep to use
    """
    # compute current updates
    di = -dt * p.tau_syn_inv * state.i
    i_new = state.i + di + p.w_in * input_tensor

    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + p.r * i_new)
    v_decayed = state.v + dv

    # compute new spikes
    z_new = threshold(v_decayed - p.v_th, p.method, p.alpha)
    # compute reset
    v_new = p.reset_method(z_new, v_decayed, p.v_reset, p.v_th)

    return z_new, CubaLIFFeedForwardState(v=v_new, i=i_new)
