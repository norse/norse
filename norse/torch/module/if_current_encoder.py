import torch

from ..functional.encode import constant_current_lif_encode
from ..functional.lif import LIFParameters


class IFConstantCurrentEncoder(torch.nn.Module):
    def __init__(
        self,
        seq_length,
        tau_mem_inv=1.0 / 1e-2,
        v_th=1.0,
        v_reset=0.0,
        dt: float = 0.001,
    ):
        super(IFConstantCurrentEncoder, self).__init__()
        self.seq_length = seq_length
        self.tau_mem_inv = tau_mem_inv
        self.v_th = v_th
        self.v_reset = v_reset
        self.dt = dt

    def forward(self, x):
        lif_parameters = LIFParameters(
            tau_mem_inv=self.tau_mem_inv, v_th=self.v_th, v_reset=self.v_reset
        )
        return constant_current_lif_encode(x, self.v, p=lif_parameters, dt=self.dt)
