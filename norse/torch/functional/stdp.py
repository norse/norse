import torch

from typing import Tuple, NamedTuple, Union

from .heaviside import heaviside


class STDPParameters(NamedTuple):
    """Parameters of spike-timing-dependent plasticity (STDP)

    Parameters:
        a_pre (float): contribution of presynaptic spikes to trace
        a_post (float): contribution of postsynaptic spikes to trace
        tau_pre_inv (float): inverse decay time constant of presynaptic spike trace in 1/ms
        tau_post_inv (float): inverse decay time constant of postsynaptic spike trace in 1/ms
        w_min (float): lower bound on synaptic weight (should be < w_max)
        w_max (float): upper bound on synaptic weight (should be > w_min)
        eta_plus (float): contribution of pre-post activity to synaptic weight (should be >= 0)
        eta_minus (float): contribution of post-pre activity to synaptic weight (should be >= 0)
    """

    a_pre: torch.Tensor = torch.as_tensor(1.0)
    a_post: torch.Tensor = torch.as_tensor(1.0)
    tau_pre_inv: torch.Tensor = torch.as_tensor(1.0 / 100e-3)
    tau_post_inv: torch.Tensor = torch.as_tensor(1.0 / 100e-3)
    w_min: float = 0.0
    w_max: float = 1.0
    eta_plus: float = 1.0
    eta_minus: float = 1.0


class STDPConvParameters(NamedTuple):
    """Parameters of spike-timing-dependent plasticity (STDP). Same as STDPParameters except
    additional information on convolution (e.g., stride, padding, dilation)

    Parameters:
        a_pre (float): contribution of presynaptic spikes to trace
        a_post (float): contribution of postsynaptic spikes to trace
        tau_pre_inv (float): inverse decay time constant of presynaptic spike trace in 1/ms
        tau_post_inv (float): inverse decay time constant of postsynaptic spike trace in 1/ms
        w_min (float): lower bound on synaptic weight (should be < w_max)
        w_max (float): upper bound on synaptic weight (should be > w_min)
        eta_plus (float): contribution of pre-post activity to synaptic weight (should be >= 0)
        eta_minus (float): contribution of post-pre activity to synaptic weight (should be >= 0)
        stride (int or tuple): convolutional stride, defaults to 1 like nn.Conv2d
        padding (int or tuple): convolutional padding, defaults to 0 like nn.Conv2d
        dilation (int or tuple): convolutional dilation, defaults to 1 like nn.Conv2d
    """

    a_pre: torch.Tensor = torch.as_tensor(1.0)
    a_post: torch.Tensor = torch.as_tensor(1.0)
    tau_pre_inv: torch.Tensor = torch.as_tensor(1.0 / 100e-3)
    tau_post_inv: torch.Tensor = torch.as_tensor(1.0 / 100e-3)
    w_min: float = 0.0
    w_max: float = 1.0
    eta_plus: float = 1.0
    eta_minus: float = 1.0
    stride: Union[int, Tuple] = 1
    padding: Union[int, Tuple] = 0
    dilation: Union[int, Tuple] = 1


class STDPState(NamedTuple):
    """State of spike-timing-dependent plasticity (STDP)

    Parameters:
        x (torch.Tensor): presynaptic spike trace
        y (torch.Tensor): postsynaptic spike trace
    """

    x: torch.Tensor
    y: torch.Tensor


def _A_plus_soft(w, w_max, eta_plus):
    """Soft upper bound for multiplicative STDP

    Parameters:
        w (torch.Tensor): synaptic weight
        w_max (float): upper bound on synaptic weight (should be > w_min)
        eta_plus (float): contribution of pre-post activity to synaptic weight (should be >= 0)
    """
    # ReLU to ensure weights outside [w_min, w_max] move towards it (when wrongly initialized)
    return torch.nn.functional.relu(w_max - w) * eta_plus


def _A_plus_hard(w, w_max, eta_plus):
    """Hard upper bound for multiplicative STDP

    Parameters:
        w (torch.Tensor): synaptic weight
        w_max (float): upper bound on synaptic weight (should be > w_min)
        eta_plus (float): contribution of pre-post activity to synaptic weight (should be >= 0)
    """
    return heaviside(w_max - w) * eta_plus


def _A_minus_soft(w, w_min, eta_minus):
    """Soft lower bound for multiplicative STDP

    Parameters:
        w (torch.Tensor): synaptic weight
        w_min (float): lower bound on synaptic weight (should be < w_max)
        eta_minus (float): contribution of post-pre activity to synaptic weight (should be >= 0)
    """
    # ReLU to ensure weights outside [w_min, w_max] move towards it (when wrongly initialized)
    return torch.nn.functional.relu(w - w_min) * eta_minus


def _A_minus_hard(w, w_min, eta_minus):
    """Hard lower bound for multiplicative STDP

    Parameters:
        w (torch.Tensor): synaptic weight
        w_min (float): lower bound on synaptic weight (should be < w_max)
        eta_minus (float): contribution of post-pre activity to synaptic weight (should be >= 0)
    """
    return heaviside(w - w_min) * eta_minus


def linear_soft_multiplicative_stdp_step(
    w: torch.Tensor,
    z_pre: torch.Tensor,
    z_post: torch.Tensor,
    state: STDPState,
    p: STDPParameters = STDPParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, STDPState]:
    """Soft-bound multiplicative STDP step for nn.Linear layer, meaning change in
    synaptic weight is dependent on current synaptic weight. Based on online implementation
    from http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity

    Parameters:
        w (torch.Tensor): synaptic weight
        z_pre (torch.Tensor): presynaptic spikes
        z_post (torch.Tensor): postsynaptic spikes
        state (STDPState): STDP state
        p (STDPParameters): STDP parameters
        dt (float): integration time step
    """
    dx = dt * p.tau_pre_inv * (-state.x + p.a_pre * z_pre)
    x_decayed = state.x + dx

    dy = dt * p.tau_post_inv * (-state.y + p.a_post * z_post)
    y_decayed = state.y + dy

    # w has shape (post, pre), so spikes/trace for postsynaptic neuron have to be transposed
    # Pre- and postsynaptic neurons can be any 2-4D (BCHW) shape
    dw_plus = _A_plus_soft(w, p.w_max, p.eta_plus) * x_decayed * z_post[..., None]
    dw_minus = _A_minus_soft(w, p.w_min, p.eta_minus) * y_decayed[..., None] * z_pre

    dw = (dw_plus - dw_minus).sum(0)

    return dw, STDPState(x=x_decayed, y=y_decayed)


def linear_hard_multiplicative_stdp_step(
    w: torch.Tensor,
    z_pre: torch.Tensor,
    z_post: torch.Tensor,
    state: STDPState,
    p: STDPParameters = STDPParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, STDPState]:
    """Hard-bound multiplicative STDP step for nn.Linear layer, meaning change in
    synaptic weight is dependent on current synaptic weight. Based on online implementation
    from http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity

    Parameters:
        w (torch.Tensor): synaptic weight
        z_pre (torch.Tensor): presynaptic spikes
        z_post (torch.Tensor): postsynaptic spikes
        state (STDPState): STDP state
        p (STDPParameters): STDP parameters
        dt (float): integration time step
    """
    dx = dt * p.tau_pre_inv * (-state.x + p.a_pre * z_pre)
    x_decayed = state.x + dx

    dy = dt * p.tau_post_inv * (-state.y + p.a_post * z_post)
    y_decayed = state.y + dy

    # w has shape (post, pre), so spikes/trace for postsynaptic neuron have to be transposed
    # Pre- and postsynaptic neurons can be any 2-4D (BCHW) shape
    dw_plus = _A_plus_hard(w, p.w_max, p.eta_plus) * x_decayed * z_post[..., None]
    dw_minus = _A_minus_hard(w, p.w_min, p.eta_minus) * y_decayed[..., None] * z_pre

    dw = (dw_plus - dw_minus).sum(0)

    return dw, STDPState(x=x_decayed, y=y_decayed)


def linear_additive_stdp_step(
    w: torch.Tensor,
    z_pre: torch.Tensor,
    z_post: torch.Tensor,
    state: STDPState,
    p: STDPParameters = STDPParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, STDPState]:
    """Additive STDP step for nn.Linear layer. Based on online implementation
    from http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity

    Parameters:
        w (torch.Tensor): synaptic weight
        z_pre (torch.Tensor): presynaptic spikes
        z_post (torch.Tensor): postsynaptic spikes
        state (STDPState): STDP state
        p (STDPParameters): STDP parameters
        dt (float): integration time step
    """
    dx = dt * p.tau_pre_inv * (-state.x + p.a_pre * z_pre)
    x_decayed = state.x + dx

    dy = dt * p.tau_post_inv * (-state.y + p.a_post * z_post)
    y_decayed = state.y + dy

    # w has shape (post, pre), so spikes/trace for postsynaptic neuron have to be transposed
    # Pre- and postsynaptic neurons can be any 2-4D (BCHW) shape
    dw_plus = p.eta_plus * x_decayed * z_post[..., None]
    dw_minus = p.eta_minus * y_decayed[..., None] * z_pre

    dw = (dw_plus - dw_minus).sum(0)

    return dw, STDPState(x=x_decayed, y=y_decayed)


def conv2d_soft_multiplicative_stdp_step(
    w: torch.Tensor,
    z_pre: torch.Tensor,
    z_post: torch.Tensor,
    state: STDPState,
    p: STDPConvParameters = STDPConvParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, STDPState]:
    """Soft-bound multiplicative STDP step for nn.Conv2d layer, meaning change in
    synaptic weight is dependent on current synaptic weight. Based on online implementation
    from http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity

    Parameters:
        w (torch.Tensor): synaptic weight
        z_pre (torch.Tensor): presynaptic spikes
        z_post (torch.Tensor): postsynaptic spikes
        state (STDPState): STDP state
        p (STDPConvParameters): STDP and convolutional parameters
        dt (float): integration time step
    """
    dx = dt * p.tau_pre_inv * (-state.x + p.a_pre * z_pre)
    x_decayed = state.x + dx

    dy = dt * p.tau_post_inv * (-state.y + p.a_post * z_post)
    y_decayed = state.y + dy

    # Get shape information
    batch_size = z_pre.shape[0]
    out_channels, _, kernel_h, kernel_w = w.shape

    # Unfold presynaptic neurons to get receptive field per postsynaptic neuron
    x_decayed_uf = torch.nn.functional.unfold(
        x_decayed,
        (kernel_h, kernel_w),
        dilation=p.dilation,
        padding=p.padding,
        stride=p.stride,
    )
    z_pre_uf = torch.nn.functional.unfold(
        z_pre,
        (kernel_h, kernel_w),
        dilation=p.dilation,
        padding=p.padding,
        stride=p.stride,
    )

    # Sum before reshape
    dw_plus = _A_plus_soft(w, p.w_max, p.eta_plus) * torch.bmm(
        z_post.view(batch_size, out_channels, -1), x_decayed_uf.permute((0, 2, 1))
    ).sum(0).view(w.shape)
    dw_minus = _A_minus_soft(w, p.w_min, p.eta_minus) * torch.bmm(
        y_decayed.view(batch_size, out_channels, -1), z_pre_uf.permute((0, 2, 1))
    ).sum(0).view(w.shape)

    dw = dw_plus - dw_minus

    return dw, STDPState(x=x_decayed, y=y_decayed)


def conv2d_hard_multiplicative_stdp_step(
    w: torch.Tensor,
    z_pre: torch.Tensor,
    z_post: torch.Tensor,
    state: STDPState,
    p: STDPConvParameters = STDPConvParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, STDPState]:
    """Hard-bound multiplicative STDP step for nn.Conv2d layer, meaning change in
    synaptic weight is dependent on current synaptic weight. Based on online implementation
    from http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity

    Parameters:
        w (torch.Tensor): synaptic weight
        z_pre (torch.Tensor): presynaptic spikes
        z_post (torch.Tensor): postsynaptic spikes
        state (STDPState): STDP state
        p (STDPConvParameters): STDP and convolutional parameters
        dt (float): integration time step
    """
    dx = dt * p.tau_pre_inv * (-state.x + p.a_pre * z_pre)
    x_decayed = state.x + dx

    dy = dt * p.tau_post_inv * (-state.y + p.a_post * z_post)
    y_decayed = state.y + dy

    # Get shape information
    batch_size = z_pre.shape[0]
    out_channels, _, kernel_h, kernel_w = w.shape

    # Unfold presynaptic neurons to get receptive field per postsynaptic neuron
    x_decayed_uf = torch.nn.functional.unfold(
        x_decayed,
        (kernel_h, kernel_w),
        dilation=p.dilation,
        padding=p.padding,
        stride=p.stride,
    )
    z_pre_uf = torch.nn.functional.unfold(
        z_pre,
        (kernel_h, kernel_w),
        dilation=p.dilation,
        padding=p.padding,
        stride=p.stride,
    )

    # Sum before reshape
    dw_plus = _A_plus_hard(w, p.w_max, p.eta_plus) * torch.bmm(
        z_post.view(batch_size, out_channels, -1), x_decayed_uf.permute((0, 2, 1))
    ).sum(0).view(w.shape)
    dw_minus = _A_minus_hard(w, p.w_min, p.eta_minus) * torch.bmm(
        y_decayed.view(batch_size, out_channels, -1), z_pre_uf.permute((0, 2, 1))
    ).sum(0).view(w.shape)

    dw = dw_plus - dw_minus

    return dw, STDPState(x=x_decayed, y=y_decayed)


def conv2d_additive_stdp_step(
    w: torch.Tensor,
    z_pre: torch.Tensor,
    z_post: torch.Tensor,
    state: STDPState,
    p: STDPConvParameters = STDPConvParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, STDPState]:
    """Additive STDP step for nn.Conv2d layer. Based on online implementation
    from http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity

    Parameters:
        w (torch.Tensor): synaptic weight
        z_pre (torch.Tensor): presynaptic spikes
        z_post (torch.Tensor): postsynaptic spikes
        state (STDPState): STDP state
        p (STDPConvParameters): STDP and convolutional parameters
        dt (float): integration time step
    """
    dx = dt * p.tau_pre_inv * (-state.x + p.a_pre * z_pre)
    x_decayed = state.x + dx

    dy = dt * p.tau_post_inv * (-state.y + p.a_post * z_post)
    y_decayed = state.y + dy

    # Get shape information
    batch_size = z_pre.shape[0]
    out_channels, _, kernel_h, kernel_w = w.shape

    # Unfold presynaptic neurons to get receptive field per postsynaptic neuron
    x_decayed_uf = torch.nn.functional.unfold(
        x_decayed,
        (kernel_h, kernel_w),
        dilation=p.dilation,
        padding=p.padding,
        stride=p.stride,
    )
    z_pre_uf = torch.nn.functional.unfold(
        z_pre,
        (kernel_h, kernel_w),
        dilation=p.dilation,
        padding=p.padding,
        stride=p.stride,
    )

    dw_plus = p.eta_plus * torch.bmm(
        z_post.view(batch_size, out_channels, -1), x_decayed_uf.permute((0, 2, 1))
    )
    dw_minus = p.eta_minus * torch.bmm(
        y_decayed.view(batch_size, out_channels, -1), z_pre_uf.permute((0, 2, 1))
    )

    dw = (dw_plus - dw_minus).sum(0).view(w.shape)

    return dw, STDPState(x=x_decayed, y=y_decayed)
