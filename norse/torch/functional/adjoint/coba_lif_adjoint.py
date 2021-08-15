import torch

from ..coba_lif import CobaLIFState, CobaLIFParameters, coba_lif_step

from typing import Tuple


class CobaLIFAdjointFunction(torch.nn.Function):
    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        z: torch.Tensor,
        v: torch.Tensor,
        g_e: torch.Tensor,
        g_i: torch.Tensor,
        input_weights: torch.Tensor,
        recurrent_weights: torch.Tensor,
        p: CobaLIFParameters,
        dt: float = 0.001,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.dt = dt
        ctx.p = p

        s = CobaLIFState(z, v, g_e, g_i)
        z_new, s_new = coba_lif_step(
            input_tensor, s, input_weights, recurrent_weights, p, dt
        )

        # dv before spiking
        dv_m = p.c_m_inv * (
            p.g_l * (p.v_rest - s.v)
            + s.g_e * (p.e_rev_E - s.v)
            + s.g_i * (p.e_rev_I - s.v)
        )
        # dv after spiking
        dv_p = p.c_m_inv * (
            p.g_l * (p.v_rest - s_new.v)
            + s.g_e * (p.e_rev_E - s_new.v)
            + s.g_i * (p.e_rev_I - s_new.v)
        )

        ctx.save_for_backward(
            input_tensor,
            s_new.z,
            s_new.v,
            s_new.g_e,
            s_new.g_i,
            dv_m,
            dv_p,
            input_weights,
            recurrent_weights,
        )
        return z_new, s_new.v, s_new.g_e, s_new.g_i

    @staticmethod
    def backward(ctx, doutput: torch.Tensor, lambda_v, lambda_g_e, lambda_g_i):
        (
            input,
            z,
            v,
            g_e,
            g_i,
            dv_m,
            dv_p,
            input_weights,
            recurrent_weights,
        ) = ctx.saved_tensors

        p = ctx.p
        dt = ctx.dt

        dw_input = torch.where(
            input_weights > 0, lambda_g_e.t().mm(input), lambda_g_i.t().mm(input)
        )
        dw_rec = torch.where(
            recurrent_weights > 0, lambda_g_e.t().mm(z), lambda_g_i.t().mm(input)
        )

        # lambda_g_e decay
        dlambda_g_e = -p.tau_syn_exc_inv * (v * lambda_g_e)
        lambda_g_e = lambda_g_e + dt * dlambda_g_e

        # lambda_g_i decay
        dlambda_g_i = -p.tau_syn_inh_inv * (v * lambda_g_i)
        lambda_g_i = lambda_g_i + dt * dlambda_g_i

        # lambda_v decay
        dlambda_v = p.c_m_inv * (-p.g_l * lambda_v - g_e * lambda_v - g_i * lambda_v)
        lambda_v = lambda_v + dt * dlambda_v

        v_output_term = torch.where(
            dv_m != 0, z * (1 / dv_m) * (doutput), torch.zeros_like(z)
        )
        v_jump_term = torch.where(dv_m != 0, z * (dv_p / dv_m), torch.zeros_like(z))
        lambda_v = (1 - z) * lambda_v + v_jump_term * lambda_v + v_output_term

        dinput = torch.where(
            input_weights > 0,
            lambda_g_e.mm(input_weights),
            lambda_g_i.mm(input_weights),
        )
        drecurrent = torch.where(
            recurrent_weights > 0,
            lambda_g_e.mm(recurrent_weights),
            lambda_g_i.mm(recurrent_weights),
        )

        return (
            dinput,
            drecurrent,
            lambda_v,
            lambda_g_e,
            lambda_g_i,
            dw_input,
            dw_rec,
            None,
            None,
        )


def coba_lif_adjoint_step(
    input: torch.Tensor,
    s: CobaLIFState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: CobaLIFParameters = CobaLIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, CobaLIFState]:
    """Implements the adjoint equations for conductance based
    leaky integrate and fire neuron.
    """
    z, v, g_e, g_i = CobaLIFAdjointFunction.apply(
        input, s.z, s.v, s.g_e, s.g_i, input_weights, recurrent_weights, p, dt
    )
    return z, CobaLIFState(z, v, g_e, g_i)
