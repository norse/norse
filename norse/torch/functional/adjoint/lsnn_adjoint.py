from typing import Tuple

import torch
import torch.jit

from norse.torch.functional.lsnn import (
    LSNNState,
    LSNNFeedForwardState,
    LSNNParameters,
    lsnn_step,
    lsnn_feed_forward_step,
)


class LSNNAdjointFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        z: torch.Tensor,
        v: torch.Tensor,
        i: torch.Tensor,
        b: torch.Tensor,
        input_weights: torch.Tensor,
        recurrent_weights: torch.Tensor,
        p: LSNNParameters = LSNNParameters(),
        dt: float = 0.001,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.p = p
        ctx.dt = dt

        s = LSNNState(z, v, i, b)
        z_new, s_new = lsnn_step(
            input_tensor, s, input_weights, recurrent_weights, p, dt
        )

        dv_m = p.tau_mem_inv * ((p.v_leak - s.v) + s.i)  # dv before spiking
        dv_p = p.tau_mem_inv * ((p.v_leak - s_new.v) + s.i)  # dv after spiking
        db_m = p.tau_adapt_inv * (p.v_th - s.b)  # db before spiking
        db_p = p.tau_adapt_inv * (p.v_th - s_new.b)  # db after spiking

        ctx.save_for_backward(
            input_tensor,
            z_new,
            dv_m,
            dv_p,
            db_m,
            db_p,
            input_weights,
            recurrent_weights,
        )
        return z_new, s_new.v, s_new.i, s_new.b

    @staticmethod
    def backward(ctx, doutput, lambda_v, lambda_i, lambda_b):
        (
            input_tensor,
            z,
            dv_m,
            dv_p,
            db_m,
            db_p,
            input_weights,
            recurrent_weights,
        ) = ctx.saved_tensors
        p = ctx.p
        dt = ctx.dt
        tau_syn_inv = p.tau_syn_inv
        tau_mem_inv = p.tau_mem_inv
        tau_adapt_inv = p.tau_adapt_inv

        dw_input = lambda_i.t().mm(input_tensor)
        dw_rec = lambda_i.t().mm(z)

        # lambda_i decay
        dlambda_i = tau_syn_inv * (lambda_v - lambda_i)
        lambda_i = lambda_i + dt * dlambda_i

        # lambda_v decay
        lambda_v = lambda_v - tau_mem_inv * dt * lambda_v
        # lambda_b decay
        lambda_b = lambda_b - tau_adapt_inv * dt * lambda_b

        v_output_term = torch.where(
            dv_m != 0, z * (1 / dv_m) * (doutput), torch.zeros_like(z)
        )
        v_jump_term = torch.where(dv_m != 0, z * (dv_p / dv_m), torch.zeros_like(z))

        b_output_term = torch.where(
            db_m != 0, z * (1 / db_m) * (doutput), torch.zeros_like(z)
        )
        b_jump_term = torch.where(db_m != 0, z * (db_p / db_m), torch.zeros_like(z))

        lambda_v = (1 - z) * lambda_v + v_jump_term * lambda_v + v_output_term
        lambda_b = (1 - z) * lambda_b + b_jump_term * lambda_b + b_output_term

        dinput = lambda_i.mm(input_weights)
        drecurrent = lambda_i.mm(recurrent_weights)
        return (
            dinput,
            drecurrent,
            lambda_v,
            lambda_i,
            lambda_b,
            dw_input,
            dw_rec,
            None,
            None,
        )


def lsnn_adjoint_step(
    input: torch.Tensor,
    s: LSNNState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: LSNNParameters = LSNNParameters(),
    dt: float = 0.001,
):
    """Implementes a single euler forward and adjoint backward
    step of a lif neuron with adaptive threshhold and current based
    exponential synapses.

    Parameters:
        input (torch.Tensor): input spikes from other cells
        s (LSNNState): current state of the LSNN unit
        input_weights (torch.Tensor): synaptic weights for input spikes
        recurrent_weights (torch.Tensor): recurrent weights for recurrent spikes
        p (LSNNParameters): parameters of the LSNN unit
        dt (torch.Tensor): integration timestep
    """
    z, v, i, b = LSNNAdjointFunction.apply(
        input, s.z, s.v, s.i, s.b, input_weights, recurrent_weights, p, dt
    )
    return z, LSNNState(z, v, i, b)


class LSNNFeedForwardAdjointFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        v: torch.Tensor,
        i: torch.Tensor,
        b: torch.Tensor,
        p: LSNNParameters = LSNNParameters(),
        dt: float = 0.001,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.p = p
        ctx.dt = dt

        s = LSNNFeedForwardState(v, i, b)
        z_new, s_new = lsnn_feed_forward_step(input, s, p, dt)

        # dv before spiking
        dv_m = p.tau_mem_inv * ((p.v_leak - s.v) + s.i)
        # dv after spiking
        dv_p = p.tau_mem_inv * ((p.v_leak - s_new.v) + s.i)
        # db before spiking
        db_m = p.tau_adapt_inv * (p.v_th - s.b)
        # db after spiking
        db_p = p.tau_adapt_inv * (p.v_th - s_new.b)

        ctx.save_for_backward(z_new, dv_m, dv_p, db_m, db_p)
        return z_new, s_new.v, s_new.i, s_new.b

    @staticmethod
    def backward(
        ctx,
        doutput: torch.Tensor,
        lambda_v: torch.Tensor,
        lambda_i: torch.Tensor,
        lambda_b: torch.Tensor,
    ):
        z, dv_m, dv_p, db_m, db_p = ctx.saved_tensors
        p = ctx.p
        dt = ctx.dt

        # lambda_i decay
        dlambda_i = p.tau_syn_inv * (lambda_v - lambda_i)
        lambda_i = lambda_i + dt * dlambda_i

        # lambda_v decay
        lambda_v = lambda_v - p.tau_mem_inv * dt * lambda_v
        # lambda_b decay
        lambda_b = lambda_b - p.tau_adapt_inv * dt * lambda_b

        v_output_term = torch.where(
            dv_m != 0, z * (1 / dv_m) * doutput, torch.zeros_like(z)
        )
        v_jump_term = torch.where(dv_m != 0, z * (dv_p / dv_m), torch.zeros_like(z))

        b_output_term = torch.where(
            db_m != 0, z * (1 / db_m) * doutput, torch.zeros_like(z)
        )
        b_jump_term = torch.where(db_m != 0, z * (db_p / db_m), torch.zeros_like(z))

        lambda_v = (1 - z) * lambda_v + v_jump_term * lambda_v + v_output_term
        lambda_b = (1 - z) * lambda_b + b_jump_term * lambda_b + b_output_term

        dinput = lambda_i

        return (dinput, lambda_v, lambda_i, lambda_b, None, None)


def lsnn_feed_forward_adjoint_step(
    input: torch.Tensor,
    s: LSNNFeedForwardState,
    p: LSNNParameters = LSNNParameters(),
    dt: float = 0.001,
):
    """Implementes a single euler forward and adjoint backward
    step of a lif neuron with adaptive threshhold and current based
    exponential synapses.

    Parameters:
        input (torch.Tensor): input spikes from other cells
        v (torch.Tensor): membrane voltage state of this cell
        i (torch.Tensor): synaptic input current state of this cell
        b (torch.Tensor): state of the adaptation variable
        input_weights (torch.Tensor): synaptic weights for input spikes
        recurrent_weights (torch.Tensor): recurrent weights for recurrent spikes
        p (LSNNParameters): parameters to use for the lsnn unit
        dt (torch.Tensor): integration timestep
    """
    z, v, i, b = LSNNFeedForwardAdjointFunction.apply(input, s.v, s.i, s.b, p, dt)
    return z, LSNNFeedForwardState(v, i, b)
