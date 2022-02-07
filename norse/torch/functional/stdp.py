from typing import Tuple

import torch

from norse.torch.functional.heaviside import heaviside


class STDPState:
    """State of spike-timing-dependent plasticity (STDP).
    Parameters:
        t_pre (torch.Tensor): presynaptic spike trace
        t_post (torch.Tensor): postsynaptic spike trace
    """

    def __init__(self, t_pre: torch.Tensor, t_post: torch.Tensor):
        self.t_pre = t_pre
        self.t_post = t_post

    def decay(
        self,
        z_pre: torch.Tensor,
        z_post: torch.Tensor,
        tau_pre_inv: torch.Tensor,
        tau_post_inv: torch.Tensor,
        a_pre: torch.Tensor,
        a_post: torch.Tensor,
        dt: float = 0.001,
    ):
        """Decay function for STDP traces.
        Parameters:
            z_pre (torch.Tensor): presynaptic spikes
            z_post (torch.Tensor): postsynaptic spikes
            tau_pre_inv (torch.Tensor): inverse time-constant for the presynaptic trace
            tau_post (torch.Tensor): inverse time-constant for the postsynaptic trace
            a_pre (torch.Tensor): presynaptic trace
            a_post (torch.Tensor): postsynaptic trace
            dt (float): time-resolution
        """
        self.t_pre = self.t_pre + (dt * tau_pre_inv * (-self.t_pre + a_pre * z_pre))
        self.t_post = self.t_post + (
            dt * tau_post_inv * (-self.t_post + a_post * z_post)
        )


class STDPParameters:
    """STDP parameters.
    Parameters:
        a_pre (torch.Tensor): Contribution of presynaptic spikes to trace
        a_post (torch.Tensor): Contribution of postsynaptic spikes to trace
        tau_pre_inv (torch.Tensor): Inverse decay time constant of presynaptic spike trace in 1/s
        tau_post_inv (torch.Tensor): Inverse decay time constant of postsynaptic spike trace in 1/s
        w_min (torch.Tensor): Lower bound on synaptic weights (should be < w_max)
        w_max (torch.Tensor): Upper bound on synaptic weight (should be > w_min)
        eta_plus (torch.Tensor): Learning rate for synaptic potentiation (0 < eta_plus << 1)
        eta_minus (torch.Tensor): Learning rate for synaptic depression (0 < eta_minus << 1)
        stdp_algorithm (string): Algorithm for STDP updates. Options in {"additive","additive_step","multiplicative_pow","multiplicative_relu"}
        mu (torch.Tensor): Exponent for multiplicative STDP (0 <= mu <= 1)
        hardbound (boolean): Impose hardbounds by clipping (recommended unles eta_* << 1)
        convolutional (boolean): Convolutional weighting kernel
        stride (int): Stride for convolution
        padding (int): Padding for convolution
        dilation (int): Dilation for convolution
    """

    def __init__(
        self,
        a_pre: torch.Tensor = torch.as_tensor(1.0),
        a_post: torch.Tensor = torch.as_tensor(1.0),
        tau_pre_inv: torch.Tensor = torch.as_tensor(1.0 / 50e-3),
        tau_post_inv: torch.Tensor = torch.as_tensor(1.0 / 50e-3),
        w_min: torch.Tensor = 0.0,
        w_max: torch.Tensor = 1.0,
        eta_plus: torch.Tensor = 1e-3,
        eta_minus: torch.Tensor = 1e-3,
        stdp_algorithm: str = "additive",
        mu: torch.Tensor = 0.0,
        hardbound: bool = True,
        convolutional: bool = False,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
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
        if self.stdp_algorithm == "additive":
            self.mu = torch.tensor(0.0)
            self.A_plus = lambda w: self.eta_plus
            self.A_minus = lambda w: self.eta_minus
        elif self.stdp_algorithm == "additive_step":
            self.mu = torch.tensor(0.0)
            self.A_plus = lambda w: self.eta_plus * heaviside(self.w_max - w)
            self.A_minus = lambda w: self.eta_minus * heaviside(w - self.w_min)
        elif self.stdp_algorithm == "multiplicative_pow":
            self.mu = torch.tensor(mu)
            self.A_plus = lambda w: self.eta_plus * torch.pow(self.w_max - w, self.mu)
            self.A_minus = lambda w: self.eta_minus * torch.pow(w - self.w_min, self.mu)
        elif self.stdp_algorithm == "multiplicative_relu":
            self.mu = torch.tensor(1.0)
            self.A_plus = lambda w: self.eta_plus * torch.nn.functional.relu(
                self.w_max - w
            )
            self.A_minus = lambda w: self.eta_minus * torch.nn.functional.relu(
                w - self.w_min
            )

        # Hard bounds
        self.hardbound = hardbound
        if self.hardbound:
            self.bounding_func = lambda w: torch.clamp(w, w_min, w_max)
        else:
            self.bounding_func = lambda w: w

        # Conv2D
        self.convolutional = convolutional
        if self.convolutional:
            self.stride = stride
            self.padding = padding
            self.dilation = dilation


def stdp_step_linear(
    z_pre: torch.Tensor,
    z_post: torch.Tensor,
    w: torch.Tensor,
    state_stdp: STDPState,
    p_stdp: STDPParameters = STDPParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, STDPState]:
    """STDP step for a FF LIF layer.
    Input:
        z_pre (torch.Tensor): Presynaptic activity z: {0,1} -> {no spike, spike}
        z_post (torch.Tensor): Postsynaptic activity z: {0,1} -> {no spike, spike}
        w (torch.Tensor): Weight tensor connecting the pre- and postsynaptic layers
        state_stdp (STDPState): STDP state
        p_stdp (STDPParameters): Parameters of STDP
        dt (float): Time-resolution
    Output:
        w (torch.tensor): Updated synaptic weights
        state_stdp (STDPState): Updated STDP state
    """

    # Update STDP traces
    state_stdp.decay(
        z_pre,
        z_post,
        p_stdp.tau_pre_inv,
        p_stdp.tau_post_inv,
        p_stdp.a_pre,
        p_stdp.a_post,
        dt,
    )

    # STDP weight update
    dw_plus = p_stdp.A_plus(w) * torch.einsum("bi,bj->ij", z_post, state_stdp.t_pre)
    dw_minus = p_stdp.A_minus(w) * torch.einsum("bi,bj->ij", state_stdp.t_post, z_pre)

    w = w + (dw_plus - dw_minus)

    # Bound checking
    w = p_stdp.bounding_func(w)

    return (w, state_stdp)


def stdp_step_conv2d(
    z_pre: torch.Tensor,
    z_post: torch.Tensor,
    w: torch.Tensor,
    state_stdp: STDPState,
    p_stdp: STDPParameters = STDPParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, STDPState]:
    """STDP step for a conv2d LIF layer.
    Input:
        z_pre (torch.tensor): Presynaptic activity z: {0,1} -> {no spike, spike}
        z_post (torch.tensor): Postsynaptic activity z: {0,1} -> {no spike, spike}
        w (torch.Tensor): Weight tensor connecting the pre- and postsynaptic layers
        state_stdp (STDPState): STDP state
        p_stdp (STDPParameters): Parameters of STDP
        dt (float): Time-resolution
    Output:
        w (torch.tensor): Updated synaptic weights
        state_stdp (STDPState): Updated STDP state
    """

    # Update STDP traces
    state_stdp.decay(
        z_pre,
        z_post,
        p_stdp.tau_pre_inv,
        p_stdp.tau_post_inv,
        p_stdp.a_pre,
        p_stdp.a_post,
        dt,
    )

    # Unfolding the convolution
    # Get shape information
    batch_size = z_pre.shape[0]
    out_channels, _, kernel_h, kernel_w = w.shape

    # Unfold presynaptic trace
    state_stdp_t_pre_uf = torch.nn.functional.unfold(
        state_stdp.t_pre,
        (kernel_h, kernel_w),
        dilation=p_stdp.dilation,
        padding=p_stdp.padding,
        stride=p_stdp.stride,
    )
    # Unfold presynaptic receptive fields
    z_pre_uf = torch.nn.functional.unfold(
        z_pre,
        (kernel_h, kernel_w),
        dilation=p_stdp.dilation,
        padding=p_stdp.padding,
        stride=p_stdp.stride,
    )

    # STDP weight update
    dw_plus = p_stdp.A_plus(w) * torch.einsum(
        "bik,bjk->ij",
        z_post.view(batch_size, out_channels, -1),
        state_stdp_t_pre_uf,
    ).view(w.shape)

    dw_minus = p_stdp.A_minus(w) * torch.einsum(
        "bik,bjk->ij",
        state_stdp.t_post.view(batch_size, out_channels, -1),
        z_pre_uf,
    ).view(w.shape)

    w = w + (dw_plus - dw_minus)

    # Bound checking
    w = p_stdp.bounding_func(w)

    return (w, state_stdp)
