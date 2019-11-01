import torch

from .heaviside import heaviside


def if_current_encoder_step(
    input_current, v, tau_mem_inv=1.0 / 1e-2, v_th=1.0, v_reset=0.0, dt=0.001
):
    dv = dt * tau_mem_inv * (input_current)
    v = v + dv
    z = heaviside(v - v_th)

    v = v - z * (v - v_reset)
    return z, v
