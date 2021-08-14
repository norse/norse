from typing import Tuple

import torch

from ..lif_refrac import LIFRefracState, LIFRefracParameters
from ..lif import LIFState
from ..lif_mc_refrac import lif_mc_refrac_step
from ..heaviside import heaviside


class LIFMCRefracAdjointFunction(torch.autograd.Function):
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
        g_coupling: torch.Tensor,
        p: LIFRefracParameters = LIFRefracParameters(),
        dt: float = 0.001,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.p = p
        ctx.dt = dt

        s = LIFRefracState(LIFState(z, v, i), rho)

        z_new, s_new = lif_mc_refrac_step(
            input_tensor, s, input_weights, recurrent_weights, g_coupling, p, dt
        )
        # dv before spiking
        dv_m = p.lif.tau_mem_inv * ((p.lif.v_leak - s.lif.v) + s.lif.i)
        # dv after spiking
        dv_p = p.lif.tau_mem_inv * ((p.lif.v_leak - s_new.lif.v) + s.lif.i)
        ctx.save_for_backward(
            input_tensor,
            s_new.lif.v,
            s_new.lif.z,
            dv_m,
            dv_p,
            input_weights,
            recurrent_weights,
            g_coupling,
            s_new.rho,
        )

        return z_new, s_new.lif.v, s_new.lif.i, s_new.rho

    @staticmethod
    def backward(ctx, doutput, lambda_v, lambda_i, lambda_rho):
        (
            input_tensor,
            v,
            z,
            dv_m,
            dv_p,
            input_weights,
            recurrent_weights,
            g_coupling,
            refrac_count,
        ) = ctx.saved_tensors
        p = ctx.p
        tau_syn_inv = p.lif.tau_syn_inv
        tau_mem_inv = p.lif.tau_mem_inv
        dt = ctx.dt

        dw_input = lambda_i.t().mm(input_tensor)
        dw_rec = lambda_i.t().mm(z)

        refrac_mask = heaviside(refrac_count)

        # update for coupling
        dg_coupling = lambda_v.t().mm(v)

        # lambda_i step
        dlambda_i = tau_syn_inv * (
            (1 - refrac_mask) * lambda_v - lambda_i
        ) + torch.linear(lambda_v, g_coupling.t())
        lambda_i = lambda_i + dt * dlambda_i

        # lambda_v decay
        lambda_v = lambda_v - tau_mem_inv * dt * (1 - refrac_mask) * lambda_v

        output_term = z * (1 / dv_m) * (doutput)
        output_term[output_term != output_term] = 0.0

        jump_term = z * (dv_p / dv_m)
        jump_term[jump_term != jump_term] = 0.0

        lambda_v = (1 - z) * lambda_v + jump_term * lambda_v + output_term

        dinput = lambda_i.mm(input_weights)
        drecurrent = lambda_i.mm(recurrent_weights)

        return (dinput, drecurrent, lambda_v, lambda_i, dw_input, dw_rec, dg_coupling)


def lif_mc_refrac_adjoint(
    input: torch.Tensor,
    s: LIFRefracState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    g_coupling: torch.Tensor,
    p: LIFRefracParameters = LIFRefracParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFRefracState]:
    z, v, i, rho = LIFMCRefracAdjointFunction.apply(
        input,
        s.lif.z,
        s.lif.v,
        s.lif.i,
        s.rho,
        input_weights,
        recurrent_weights,
        g_coupling,
        p,
        dt,
    )
    return z, LIFRefracState(LIFState(z, v, i), rho)
