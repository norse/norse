import torch

from norse.torch.functional.lif_mc import lif_mc_step
from norse.torch.functional.lif import LIFState, LIFParameters

from typing import Tuple


class LIFMCAdjointFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        z: torch.Tensor,
        v: torch.Tensor,
        i: torch.Tensor,
        input_weights: torch.Tensor,
        recurrent_weights: torch.Tensor,
        g_coupling: torch.Tensor,
        p: LIFParameters = LIFParameters(),
        dt: float = 0.001,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.dt = dt
        ctx.p = p

        s = LIFState(z, v, i)
        z_new, s_new = lif_mc_step(
            input_tensor, s, input_weights, recurrent_weights, g_coupling, p, dt
        )

        # dv before spiking
        dv_m = p.tau_mem_inv * ((p.v_leak - s.v) + s.i)
        # dv after spiking
        dv_p = p.tau_mem_inv * ((p.v_leak - s_new.v) + s.i)
        ctx.save_for_backward(
            input_tensor,
            s_new.v,
            z_new,
            dv_m,
            dv_p,
            input_weights,
            recurrent_weights,
            g_coupling,
        )

        return z_new, s_new.v, s_new.i

    @staticmethod
    def backward(ctx, doutput, lambda_v, lambda_i):
        (
            input_tensor,
            v,
            z,
            dv_m,
            dv_p,
            input_weights,
            recurrent_weights,
            g_coupling,
        ) = ctx.saved_tensors
        p = ctx.p
        dt = ctx.dt

        dw_input = lambda_i.t().mm(input_tensor)
        dw_rec = lambda_i.t().mm(z)

        # update for coupling
        dg_coupling = lambda_v.t().mm(v)

        # lambda_i step
        dlambda_i = p.tau_syn_inv * (lambda_v - lambda_i) + torch.linear(
            lambda_v, g_coupling.t()
        )
        lambda_i = lambda_i + dt * dlambda_i

        # lambda_v decay
        lambda_v = lambda_v - p.tau_mem_inv * dt * lambda_v

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
            dw_input,
            dw_rec,
            dg_coupling,
            None,
            None,
        )


def lif_mc_adjoint_step(
    input: torch.Tensor,
    s: LIFState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    g_coupling: torch.Tensor,
    p: LIFParameters = LIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFState]:
    z, v, i = LIFMCAdjointFunction.apply(
        input, s.z, s.v, s.i, input_weights, recurrent_weights, g_coupling, p, dt
    )
    return z, LIFState(z, v, i)
