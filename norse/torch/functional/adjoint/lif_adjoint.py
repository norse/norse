import torch
import torch.jit

from norse.torch.functional.lif import (
    LIFState,
    LIFFeedForwardState,
    LIFParameters,
    lif_step,
    lif_feed_forward_step,
    lif_feed_forward_step_sparse,
    lif_step_sparse,
)
from typing import Tuple


class LIFAdjointFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        z: torch.Tensor,
        v: torch.Tensor,
        i: torch.Tensor,
        input_weights: torch.Tensor,
        recurrent_weights: torch.Tensor,
        p: LIFParameters = LIFParameters(),
        dt: float = 0.001,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.tau_syn_inv = p.tau_syn_inv
        ctx.tau_mem_inv = p.tau_mem_inv
        ctx.v_th = p.v_th
        ctx.v_reset = p.v_reset
        ctx.dt = dt
        s = LIFState(z, v, i)
        z_new, s_new = lif_step(
            input_tensor, s, input_weights, recurrent_weights, p, dt
        )

        # dv before spiking
        dv_m = p.tau_mem_inv * ((p.v_leak - s.v) + s.i)
        # dv after spiking
        dv_p = p.tau_mem_inv * ((p.v_leak - s_new.v) + s.i)

        ctx.save_for_backward(
            input_tensor, z_new, dv_m, dv_p, input_weights, recurrent_weights
        )
        return z_new, s_new.v, s_new.i

    @staticmethod
    def backward(ctx, doutput, lambda_v, lambda_i):
        (
            input_tensor,
            z,
            dv_m,
            dv_p,
            input_weights,
            recurrent_weights,
        ) = ctx.saved_tensors
        tau_syn_inv = ctx.tau_syn_inv
        tau_mem_inv = ctx.tau_mem_inv
        dt = ctx.dt

        dw_input = lambda_i.t().mm(input_tensor)
        dw_rec = lambda_i.t().mm(z)

        # lambda_i decay
        dlambda_i = tau_syn_inv * (lambda_v - lambda_i)
        lambda_i = lambda_i + dt * dlambda_i

        # lambda_v decay
        lambda_v = lambda_v - tau_mem_inv * dt * lambda_v

        output_term = z * (1 / dv_m) * (doutput)
        output_term[output_term != output_term] = 0.0

        jump_term = z * (dv_p / dv_m)
        jump_term[jump_term != jump_term] = 0.0

        lambda_v = (1 - z) * lambda_v + jump_term * lambda_v + output_term

        dinput = lambda_i.mm(input_weights)
        drecurrent = lambda_i.mm(recurrent_weights)

        return (dinput, drecurrent, lambda_v, lambda_i, dw_input, dw_rec, None, None)


def lif_adjoint_step(
    input: torch.Tensor,
    s: LIFState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: LIFParameters = LIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFState]:
    """Implementes a single euler forward and adjoint backward
    step of a leaky integrate and fire neuron with current based
    exponential synapses.

    Parameters:
        input (torch.Tensor): input spikes from other cells
        s (LIFState): state of the lif neurons
        input_weights (torch.Tensor): synaptic weights for input spikes
        recurrent_weights (torch.Tensor): recurrent weights for recurrent spikes
        p (LIFParameters): parameters of the lif neurons
        dt (float): time step of integration
    """
    z, v, i = LIFAdjointFunction.apply(
        input, s.z, s.v, s.i, input_weights, recurrent_weights, p, dt
    )
    return z, LIFState(z, v, i)


class LIFSparseAdjointFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        z: torch.Tensor,
        v: torch.Tensor,
        i: torch.Tensor,
        input_weights: torch.Tensor,
        recurrent_weights: torch.Tensor,
        p: LIFParameters = LIFParameters(),
        dt: float = 0.001,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.tau_syn_inv = p.tau_syn_inv
        ctx.tau_mem_inv = p.tau_mem_inv
        ctx.v_th = p.v_th
        ctx.v_reset = p.v_reset
        ctx.dt = dt
        s = LIFState(z, v, i)
        z_new, s_new = lif_step_sparse(
            input, s, input_weights, recurrent_weights, p, dt
        )

        # dv before spiking
        dv_m = p.tau_mem_inv * ((p.v_leak - s_new.v) + s.i)
        # dv after spiking
        dv_p = p.tau_mem_inv * ((p.v_leak - s_new.v) + s.i)

        ctx.save_for_backward(
            input,
            z_new,
            dv_m.sparse_mask(z_new),
            dv_p.sparse_mask(z_new),
            input_weights,
            recurrent_weights,
        )
        return z_new, s_new.v, s_new.i

    @staticmethod
    def backward(ctx, doutput, lambda_v, lambda_i):
        input, z, dv_m, dv_p, input_weights, recurrent_weights = ctx.saved_tensors
        tau_syn_inv = ctx.tau_syn_inv
        tau_mem_inv = ctx.tau_mem_inv
        dt = ctx.dt
        dv_m = dv_m.to_dense()
        dv_p = dv_p.to_dense()
        z = z.to_dense()

        dw_input = lambda_i.t().mm(input)
        dw_rec = lambda_i.t().mm(z)

        # lambda_i decay
        dlambda_i = tau_syn_inv * (lambda_v - lambda_i)
        lambda_i = lambda_i + dt * dlambda_i

        # lambda_v decay
        lambda_v = lambda_v - tau_mem_inv * dt * lambda_v

        output_term = z * (1 / dv_m) * (doutput)
        output_term[output_term != output_term] = 0.0

        jump_term = z * (dv_p / dv_m)
        jump_term[jump_term != jump_term] = 0.0

        lambda_v = (1 - z) * lambda_v + jump_term * lambda_v + output_term

        dinput = lambda_i.mm(input_weights)
        drecurrent = lambda_i.mm(recurrent_weights)

        return (dinput, drecurrent, lambda_v, lambda_i, dw_input, dw_rec, None, None)


def lif_adjoint_step_sparse(
    input: torch.Tensor,
    s: LIFState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: LIFParameters = LIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFState]:
    """Implementes a single euler forward and adjoint backward
    step of a leaky integrate and fire neuron with current based
    exponential synapses.

    Parameters:
        input (torch.Tensor): input spikes from other cells
        s (LIFState): state of the lif neurons
        input_weights (torch.Tensor): synaptic weights for input spikes
        recurrent_weights (torch.Tensor): recurrent weights for recurrent spikes
        p (LIFParameters): parameters of the lif neurons
        dt (float): time step of integration
    """
    z, v, i = LIFSparseAdjointFunction.apply(
        input, s.z, s.v, s.i, input_weights, recurrent_weights, p, dt
    )
    return z, LIFState(z, v, i)


class LIFFeedForwardAdjointFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        v: torch.Tensor,
        i: torch.Tensor,
        p: LIFParameters = LIFParameters(),
        dt: float = 0.001,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.p = p
        ctx.dt = dt
        z_new, s_new = lif_feed_forward_step(
            input, LIFFeedForwardState(v, i), p=p, dt=dt
        )

        # dv before spiking
        dv_m = p.tau_mem_inv * ((p.v_leak - v) + i)
        # dv after spiking
        dv_p = p.tau_mem_inv * ((p.v_leak - s_new.v) + i)

        ctx.save_for_backward(z_new, dv_m, dv_p)
        return z_new, s_new.v, s_new.i

    @staticmethod
    def backward(
        ctx, doutput: torch.Tensor, lambda_v: torch.Tensor, lambda_i: torch.Tensor
    ):
        z, dv_m, dv_p = ctx.saved_tensors
        p = ctx.p
        dt = ctx.dt

        # lambda_i decay
        dlambda_i = p.tau_syn_inv * (lambda_v - lambda_i)
        lambda_i = lambda_i + dt * dlambda_i

        # lambda_v decay
        lambda_v = lambda_v - p.tau_mem_inv * dt * lambda_v

        output_term = z * (1 / dv_m) * doutput
        output_term[output_term != output_term] = 0.0

        jump_term = z * (dv_p / dv_m)
        jump_term[jump_term != jump_term] = 0.0

        lambda_v = (1 - z) * lambda_v + jump_term * lambda_v + output_term
        dinput = lambda_i

        return (dinput, lambda_v, lambda_i, None, None)


def lif_feed_forward_adjoint_step(
    input: torch.Tensor,
    s: LIFFeedForwardState,
    p: LIFParameters = LIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFFeedForwardState]:
    """Implementes a single euler forward and adjoint backward
    step of a leaky integrate and fire neuron with current based
    exponential synapses.

    Parameters:
        input (torch.Tensor): input spikes from other cells
        s (LIFFeedForwardState): state of leaky integrate and fire neuron
        p (LIFParameters): leaky integrate and fire parameters
        dt (float): time step of integration
    """
    z_new, v_new, i_new = LIFFeedForwardAdjointFunction.apply(input, s.v, s.i, p, dt)
    return z_new, LIFFeedForwardState(v_new, i_new)


class LIFFeedForwardSparseAdjointFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        v: torch.Tensor,
        i: torch.Tensor,
        p: LIFParameters = LIFParameters(),
        dt: float = 0.001,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.p = p
        ctx.dt = dt
        z_new, s_new = lif_feed_forward_step_sparse(
            input, LIFFeedForwardState(v, i), p=p, dt=dt
        )

        # dv before spiking
        dv_m = p.tau_mem_inv * ((p.v_leak - v) + i)
        # dv after spiking
        dv_p = p.tau_mem_inv * ((p.v_leak - s_new.v) + i)

        ctx.save_for_backward(z_new, dv_m.sparse_mask(z_new), dv_p.sparse_mask(z_new))
        return z_new.to_dense(), s_new.v, s_new.i

    @staticmethod
    def backward(
        ctx, doutput: torch.Tensor, lambda_v: torch.Tensor, lambda_i: torch.Tensor
    ):
        z, dv_m, dv_p = ctx.saved_tensors
        p = ctx.p
        dt = ctx.dt
        dv_m = dv_m.to_dense()
        dv_p = dv_p.to_dense()
        z = z.to_dense()

        # lambda_i decay
        dlambda_i = p.tau_syn_inv * (lambda_v - lambda_i)
        lambda_i = lambda_i + dt * dlambda_i

        # lambda_v decay
        lambda_v = lambda_v - p.tau_mem_inv * dt * lambda_v

        output_term = z * (1 / dv_m) * doutput
        output_term[output_term != output_term] = 0.0

        jump_term = z * (dv_p / dv_m)
        jump_term[jump_term != jump_term] = 0.0

        lambda_v = (1 - z) * lambda_v + jump_term * lambda_v + output_term
        dinput = lambda_i

        return (dinput, lambda_v, lambda_i, None, None)


def lif_feed_forward_adjoint_step_sparse(
    input: torch.Tensor,
    s: LIFFeedForwardState,
    p: LIFParameters = LIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFFeedForwardState]:
    """Implementes a single euler forward and adjoint backward
    step of a leaky integrate and fire neuron with current based
    exponential synapses.

    Parameters:
        input (torch.Tensor): input spikes from other cells
        s (LIFFeedForwardState): state of leaky integrate and fire neuron
        p (LIFParameters): leaky integrate and fire parameters
        dt (float): time step of integration
    """
    z_new, v_new, i_new = LIFFeedForwardSparseAdjointFunction.apply(
        input, s.v, s.i, p, dt
    )
    return z_new, LIFFeedForwardState(v_new, i_new)
