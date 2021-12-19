from typing import Tuple

import torch

from norse.torch.functional.lif_refrac import (
    LIFRefracState,
    LIFRefracFeedForwardState,
    LIFRefracParameters,
    lif_refrac_step,
    lif_refrac_step_sparse,
    lif_refrac_feed_forward_step,
)
from norse.torch.functional.lif import LIFState, LIFFeedForwardState
from norse.torch.functional.heaviside import heaviside


class LIFAdjointRefracFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        z: torch.Tensor,
        v: torch.Tensor,
        i: torch.Tensor,
        rho: torch.Tensor,
        input_weights: torch.Tensor,
        recurrent_weights: torch.Tensor,
        p: LIFRefracParameters = LIFRefracParameters(),
        dt: float = 0.001,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.p = p
        ctx.dt = dt

        s = LIFRefracState(LIFState(z, v, i), rho)
        z_new, s_new = lif_refrac_step(
            input_tensor, s, input_weights, recurrent_weights, p, dt
        )

        # dv before spiking
        dv_m = p.lif.tau_mem_inv * ((p.lif.v_leak - s.lif.v) + s.lif.i)
        # dv after spiking
        dv_p = p.lif.tau_mem_inv * ((p.lif.v_leak - s_new.lif.v) + s.lif.i)

        ctx.save_for_backward(
            input_tensor, z_new, s_new.rho, dv_m, dv_p, input_weights, recurrent_weights
        )
        return z_new, s_new.lif.v, s_new.lif.i, s_new.rho

    @staticmethod
    def backward(ctx, doutput, lambda_v, lambda_i, lambda_rho):
        (
            input_tensor,
            z,
            rho,
            dv_m,
            dv_p,
            input_weights,
            recurrent_weights,
        ) = ctx.saved_tensors
        p = ctx.p
        dt = ctx.dt

        dw_input = lambda_i.t().mm(input_tensor)
        dw_rec = lambda_i.t().mm(z)

        refrac_mask = heaviside(rho)

        # lambda_i decay
        dlambda_i = p.tau_syn_inv * ((1 - refrac_mask) * lambda_v - lambda_i)
        lambda_i = lambda_i + dt * dlambda_i

        # lambda_v decay
        lambda_v = lambda_v - (1 - refrac_mask) * p.tau_mem_inv * dt * lambda_v

        output_term = z * (1 / dv_m) * (doutput)
        output_term[output_term != output_term] = 0.0

        jump_term = z * (dv_p / dv_m)
        jump_term[jump_term != jump_term] = 0.0

        lambda_v = (1 - z) * lambda_v + jump_term * lambda_v + output_term

        dinput = lambda_i.mm(input_weights)
        drecurrent = lambda_i.mm(recurrent_weights)
        return (
            dinput,
            drecurrent,
            lambda_v,
            lambda_i,
            lambda_rho,
            dw_input,
            dw_rec,
            None,
            None,
        )


def lif_refrac_adjoint_step(
    input: torch.Tensor,
    s: LIFRefracState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: LIFRefracParameters = LIFRefracParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFRefracState]:
    """Implementes a single euler forward and adjoint backward
    step of a leaky integrate and fire neuron with current based
    exponential synapses and a refractory period.
    """
    z, v, i, rho = LIFAdjointRefracFunction.apply(
        input, s.lif.z, s.lif.v, s.lif.i, s.rho, input_weights, recurrent_weights, p, dt
    )
    return z, LIFRefracState(LIFState(z, v, i), rho)


class LIFSparseAdjointRefracFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        lif_state: LIFState,
        rho: torch.Tensor,
        input_weights: torch.Tensor,
        recurrent_weights: torch.Tensor,
        p: LIFRefracParameters = LIFRefracParameters(),
        dt: float = 0.001,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.tau_sys_inv = p.lif.tau_syn_inv
        ctx.tau_mem_inv = p.lif.tau_mem_inv
        ctx.v_th = p.lif.v_th
        ctx.v_reset = p.lif.v_reset
        ctx.rho_reset = p.rho_reset
        ctx.dt = dt
        s = LIFRefracState(lif_state, rho)
        z_new, s_new = lif_refrac_step_sparse(
            input, s, input_weights, recurrent_weights, p, dt
        )

        # dv before spiking
        dv_m = p.lif.tau_mem_inv * ((p.lif.v_leak - s_new.lif.v) + s.lif.i)
        # dev after spiking
        dv_p = p.lif.tau_mem_inv * ((p.lif.v_leak - s_new.lif.v) + s.lif.i)
        refrac_period_end = (s.rho == 1).float()
        ctx.save_for_backward(
            input,
            z_new,
            s_new.rho,
            refrac_period_end,
            dv_m.sparse_mask(z_new),
            dv_p.sparse_mask(z_new),
            input_weights,
            recurrent_weights,
        )
        return z_new, s_new.lif.v, s_new.lif.i, s_new.rho

    @staticmethod
    def backward(
        ctx,
        doutput: torch.Tensor,
        lambda_v: torch.Tensor,
        lambda_i: torch.Tensor,
        lambda_rho: torch.Tensor,
    ):
        (
            input_tensor,
            z,
            refrac_count,
            refrac_period_end,
            dv_m,
            dv_p,
            input_weights,
            recurrent_weights,
        ) = ctx.saved_tensors
        tau_syn_inv = ctx.tau_syn_inv
        tau_mem_inv = ctx.taus_mem_inv
        dt = ctx.dt
        refrac_mask = heaviside(refrac_count)
        dv_m = dv_m.to_dense()
        dv_p = dv_p.to_dense()
        z = z.to_dense()

        dw_input = lambda_i.t().mm(input_tensor)
        dw_rec = lambda_i.t().mm(z)

        # lambda_i decay
        dlambda_i = tau_syn_inv * ((1 - refrac_mask) * lambda_v - lambda_i)
        lambda_i = lambda_i + dt * dlambda_i

        # lambda_v decay
        lambda_v = lambda_v - (1 - refrac_mask) * tau_mem_inv * dt * lambda_v

        output_term = z * (1 / dv_m) * (doutput)
        output_term[output_term != output_term] = 0.0

        jump_term = z * (dv_p / dv_m)
        jump_term[jump_term != jump_term] = 0.0

        lambda_v = (1 - z) * lambda_v + jump_term * lambda_v + output_term
        lambda_rho = (
            1 - refrac_period_end
        ) * lambda_rho + refrac_period_end * lambda_v * dv_p
        dinput = lambda_i.mm(input_weights)
        drecurrent = lambda_i.mm(recurrent_weights)
        return (
            dinput,
            drecurrent,
            lambda_v,
            lambda_i,
            lambda_rho,
            dw_input,
            dw_rec,
            None,
            None,
        )


def lif_refrac_adjoint_step_sparse(
    input: torch.Tensor,
    s: LIFRefracState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: LIFRefracParameters,
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFRefracState]:
    z, v, i, rho = LIFSparseAdjointRefracFunction.apply(
        input, s.lif, s.rho, input_weights, recurrent_weights, p, dt
    )

    return z, LIFRefracState(LIFState(z, v, i), rho)


class LIFAdjointRefracFeedForwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        v: torch.Tensor,
        i: torch.Tensor,
        rho: torch.Tensor,
        p: LIFRefracParameters = LIFRefracParameters(),
        dt: float = 0.001,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.tau_syn_inv = p.lif.tau_syn_inv
        ctx.tau_mem_inv = p.lif.tau_mem_inv
        ctx.dt = dt
        s = LIFRefracFeedForwardState(LIFFeedForwardState(v, i), rho)

        z_new, s_new = lif_refrac_feed_forward_step(input, s, p, dt)

        # dv before spiking
        dv_m = p.lif.tau_mem_inv * ((p.lif.v_leak - s.lif.v) + s.lif.i)
        # dv after spiking
        dv_p = p.lif.tau_mem_inv * ((p.lif.v_leak - s_new.lif.v) + s.lif.i)

        refrac_period_end = (s.rho == 1).float()
        ctx.save_for_backward(input, z_new, s_new.rho, refrac_period_end, dv_m, dv_p)
        return z_new, s_new.lif.v, s_new.lif.i, s_new.rho

    @staticmethod
    def backward(ctx, doutput, lambda_v, lambda_i, lambda_rho):
        input, z, refrac_count, refrac_period_end, dv_m, dv_p = ctx.saved_tensors
        tau_syn_inv = ctx.tau_syn_inv
        tau_mem_inv = ctx.tau_mem_inv
        dt = ctx.dt

        refrac_mask = heaviside(refrac_count)

        # lambda_i decay
        dlambda_i = tau_syn_inv * ((1 - refrac_mask) * lambda_v - lambda_i)
        lambda_i = lambda_i + dt * dlambda_i

        # lambda_v decay
        lambda_v = lambda_v - (1 - refrac_mask) * tau_mem_inv * dt * lambda_v

        output_term = z * (1 / dv_m) * doutput
        output_term[output_term != output_term] = 0.0

        jump_term = z * (lambda_rho / dv_m)
        jump_term[jump_term != jump_term] = 0.0

        lambda_v = (1 - z) * lambda_v + jump_term + output_term
        lambda_rho = (
            1 - refrac_period_end
        ) * lambda_rho + refrac_period_end * lambda_v * dv_p

        return (lambda_i, lambda_v, lambda_i, lambda_rho, None, None)


def lif_refrac_feed_forward_adjoint_step(
    input: torch.Tensor,
    s: LIFRefracFeedForwardState,
    p: LIFRefracParameters = LIFRefracParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFRefracFeedForwardState]:
    """Implementes a single euler forward and adjoint backward
    step of a leaky integrate and fire neuron with current based
    exponential synapses and a refractory period.
    """
    z, v, i, rho = LIFAdjointRefracFeedForwardFunction.apply(
        input, s.lif.v, s.lif.i, s.rho, p, dt
    )
    return z, LIFRefracFeedForwardState(LIFFeedForwardState(v, i), rho)
