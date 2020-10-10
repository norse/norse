# Includes
from typing import Tuple

import torch
from torch import clamp, relu, pow as exponentiate

from .heaviside import heaviside
from .lif import LIFFeedForwardState, LIFParameters, lif_feed_forward_step


# STDP state class
class STDPState:
    """State of spike-timing-dependent plasticity (STDP)
    Parameters:
        t_pre (torch.Tensor): presynaptic spike trace
        t_post (torch.Tensor): postsynaptic spike trace
    """
    def __init__(self, t_pre, t_post):
        self.t_pre = t_pre
        self.t_post = t_post

    def decay(self,
        z_pre, z_post,
        tau_pre_inv, tau_post_inv,
        a_pre, a_post,
        dt,
    ):
        self.t_pre +=  ( dt * tau_pre_inv  * (-self.t_pre  + a_pre*z_pre  ) )
        self.t_post += ( dt * tau_post_inv * (-self.t_post + a_post*z_post) )

# STDP parameters class
class STDPParameters:
    """STDP parameters.
    Parameters:
        a_pre (float): Contribution of presynaptic spikes to trace
        a_post (float): Contribution of postsynaptic spikes to trace
        tau_pre_inv (float): Inverse decay time constant of presynaptic spike trace in 1/s
        tau_post_inv (float): Inverse decay time constant of postsynaptic spike trace in 1/s
        w_min (float): Lower bound on synaptic weights (should be < w_max)
        w_max (float): Upper bound on synaptic weight (should be > w_min)
        eta_plus (float): Learning rate for synaptic potentiation (0 < eta_plus << 1)
        eta_minus (float): Learning rate for synaptic depression (0 < eta_minus << 1)
        stdp_algorithm (string): Algorithm for STDP updates. Options in {"additive","additive_step","multiplicative_pow","multiplicative_relu"}
        mu (float): Exponent for multiplicative STDP (0 <= mu <= 1)
        hardbound (boolean): Impose hardbounds by clipping (recommended unles eta_* << 1)
        convolutional (boolean): Convolutional weighting kernel
        stride (int): Stride for convolution
        padding (int): Padding for convolution
        dilation (int): Dilation for convolution
    """
    def __init__(self,
        a_pre = torch.as_tensor(1.0),
        a_post = torch.as_tensor(1.0),
        tau_pre_inv = torch.as_tensor(1.0 / 50e-3),
        tau_post_inv = torch.as_tensor(1.0 / 50e-3),
        w_min = 0.0,
        w_max = 1.0,
        eta_plus = 1e-3,
        eta_minus = 1e-3,
        stdp_algorithm = "additive",
        mu = 0.0,
        hardbound = True,
        convolutional = False,
        stride = 1, padding = 0, dilation = 1,
    ):
        self.a_pre = a_pre
        self.a_post = a_post
        self.tau_pre_inv = tau_pre_inv
        self.tau_post_inv = tau_post_inv
        self.w_min = w_min
        self.w_max = w_max
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.stdp_algorithm = stdp_algorithm

        if (self.stdp_algorithm == "additive"):
            self.mu = torch.tensor(0.0)
            self.A_plus = lambda w: self.eta_plus
            self.A_minus = lambda w: self.eta_minus
        elif (self.stdp_algorithm == "additive_step"):
            self.mu = torch.tensor(0.0)
            self.A_plus = lambda w: self.eta_plus*heaviside(self.w_max - w)
            self.A_minus = lambda w: self.eta_minus*heaviside(w - self.w_min)
        elif (self.stdp_algorithm == "multiplicative_pow"):
            self.mu = torch.tensor(mu)
            self.A_plus = lambda w: self.eta_plus*exponentiate(self.w_max - w, self.mu)
            self.A_minus = lambda w: self.eta_minus*exponentiate(w - self.w_min, self.mu)
        elif (self.stdp_algorithm == "multiplicative_relu"):
            self.mu = torch.tensor(1.0)
            self.A_plus = lambda w: self.eta_plus*relu(self.w_max - w)
            self.A_minus = lambda w: self.eta_minus*relu(w - self.w_min)

        # Hard bounds
        self.hardbound = hardbound
        if (self.hardbound):
            self.bounding_func = lambda w: clamp(w, w_min, w_max)
        else:
            self.bounding_func = lambda w: w

        # Conv2D
        self.convolutional = convolutional
        if convolutional:
            self.stride = stride
            self.padding = padding
            self.dilation = dilation

# %% Linear stepper
def lif_linear_stdp_step(
    z_pre: torch.Tensor,
    w: torch.Tensor,
    state_post: LIFFeedForwardState = LIFFeedForwardState(0, 0),
    p_post: LIFParameters = LIFParameters(),
    state_stdp: STDPState = STDPState,
    p_stdp: STDPParameters = STDPParameters(),
    dt: float = 0.001,
) -> Tuple[ torch.Tensor, LIFFeedForwardState,
            torch.Tensor, STDPState ]:
    """STDP step for a FF LIF layer.
    Input:
        z_pre (torch.tensor): Presynaptic activity z: {0,1} -> {no spike, spike}
        w (torch.Tensor): Weight tensor connecting the pre- and postsynaptic layers
        state_post (LIFFeedForwardState): State of the postsynaptic layer before pre-synaptic input
        p_post (LIFParameters): Parameters of the postsynaptic layer
        state_stdp (STDPState): STDP state
        p_stdp (STDPParameters): Parameters of STDP
        dt (float): Time-resolution
    Output:
        z_post (torch.tensor): Postsynaptic activity z: {0,1} -> {no spike, spike}
        state_post (LIFFeedForwardState): Final state of the postsynaptic layer
        w (torch.tensor): Updated synaptic weights
        state_stdp (STDPState): Updated STDP state
    """

    # Update post-synaptic layer
    z_post, state_post = lif_feed_forward_step(
        torch.nn.functional.linear(z_pre, w), # Integrate into LIF FF class
        state_post, p_post, dt
    )

    # Update STDP traces
    state_stdp.decay(
        z_pre,z_post,
        p_stdp.tau_pre_inv,p_stdp.tau_post_inv,
        p_stdp.a_pre, p_stdp.a_post,
        dt
    )

    # STDP weight update
    dw_plus = p_stdp.A_plus(w) * torch.einsum(
        'bi,bj->ij',
        z_post, state_stdp.t_pre
    )
    dw_minus = p_stdp.A_minus(w) * torch.einsum(
        'bi,bj->ij',
        state_stdp.t_post, z_pre
    )

    w += (dw_plus - dw_minus)

    # Bound checking
    w = p_stdp.bounding_func(w)

    return (z_post, state_post,
            w, state_stdp
    )

# %% Conv2D stepper
def lif_conv2d_stdp_step(
    z_pre: torch.Tensor,
    w: torch.Tensor,
    state_post: LIFFeedForwardState = LIFFeedForwardState(0, 0),
    p_post: LIFParameters = LIFParameters(),
    state_stdp: STDPState = STDPState,
    p_stdp: STDPParameters = STDPParameters(),
    dt: float = 0.001,
) -> Tuple[ torch.Tensor, LIFFeedForwardState,
            torch.Tensor, STDPState ]:
    """STDP step for a conv2d LIF layer.
    Input:
        z_pre (torch.tensor): Presynaptic activity z: {0,1} -> {no spike, spike}
        w (torch.Tensor): Weight tensor connecting the pre- and postsynaptic layers
        state_post (LIFFeedForwardState): State of the postsynaptic layer before pre-synaptic input
        p_post (LIFParameters): Parameters of the postsynaptic layer
        state_stdp (STDPState): STDP state
        p_stdp (STDPParameters): Parameters of STDP
        dt (float): Time-resolution
    Output:
        z_post (torch.tensor): Postsynaptic activity z: {0,1} -> {no spike, spike}
        state_post (LIFFeedForwardState): Final state of the postsynaptic layer
        w (torch.tensor): Updated synaptic weights
        state_stdp (STDPState): Updated STDP state
    """

    # Update post-synaptic layer
    z_post, state_post = lif_feed_forward_step(
        torch.nn.functional.conv2d(
                z_pre, w,
                stride = p_stdp.stride,
                padding = p_stdp.padding,
                dilation = p_stdp.dilation,
            ), # Integrate into LIF FF class
        state_post, p_post, dt
    )

    # Update STDP traces
    state_stdp.decay(
        z_pre,z_post,
        p_stdp.tau_pre_inv,p_stdp.tau_post_inv,
        p_stdp.a_pre,p_stdp.a_post,
        dt
    )

    # Unfolding the convolution
    # Get shape information
    batch_size = z_pre.shape[0]
    out_channels, _, kernel_h, kernel_w = w.shape

    # Unfold presynaptic trace
    state_stdp_t_pre_uf = torch.nn.functional.unfold(
        state_stdp.t_pre,
        (kernel_h, kernel_w),
        dilation = p_stdp.dilation,
        padding = p_stdp.padding,
        stride = p_stdp.stride,
    )
    # Unfold presynaptic receptive fields
    z_pre_uf = torch.nn.functional.unfold(
        z_pre,
        (kernel_h, kernel_w),
        dilation = p_stdp.dilation,
        padding = p_stdp.padding,
        stride = p_stdp.stride,
    )

    # STDP weight update
    dw_plus = p_stdp.A_plus(w) * torch.einsum(
        'bik,bjk->ij',
        z_post.view(batch_size, out_channels, -1),
        state_stdp_t_pre_uf,
    ).view(w.shape)

    dw_minus = p_stdp.A_minus(w) * torch.einsum(
        'bik,bjk->ij',
        state_stdp.t_post.view(batch_size, out_channels, -1), z_pre_uf,
    ).view(w.shape)

    w += (dw_plus - dw_minus)

    # Bound checking
    w = p_stdp.bounding_func(w)

    return (z_post, state_post,
            w, state_stdp
    )
