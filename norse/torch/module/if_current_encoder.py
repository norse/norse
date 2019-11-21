import torch

from ..functional.if_current_encoder import if_current_encoder_step


class IFConstantCurrentEncoder(torch.nn.Module):
    def __init__(
        self,
        seq_length,
        tau_mem_inv=1.0 / 1e-2,
        v_th=1.0,
        v_reset=0.0,
        dt: float = 0.001,
        device="cpu",
    ):
        super(IFConstantCurrentEncoder, self).__init__()
        self.seq_length = seq_length
        self.tau_mem_inv = tau_mem_inv
        self.v_th = v_th
        self.v_reset = v_reset
        self.device = device

    def forward(self, x):
        v = torch.zeros(*x.shape, device=self.device)
        z = torch.zeros(*x.shape, device=self.device)
        voltages = torch.zeros(self.seq_length, *x.shape, device=self.device)
        spikes = torch.zeros(self.seq_length, *x.shape, device=self.device)

        for ts in range(self.seq_length):
            z, v = if_current_encoder_step(
                input_current=x,
                v=v,
                tau_mem_inv=self.tau_mem_inv,
                v_th=self.v_th,
                v_reset=self.v_reset,
            )
            voltages[ts, :, :] = v
            spikes[ts, :, :] = z
        return spikes, voltages
