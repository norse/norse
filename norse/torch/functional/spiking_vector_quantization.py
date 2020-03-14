import torch


def spiking_vector_quantization_step(v: torch.Tensor, phi: torch.Tensor):
    """Performs the quantization method explained in

    "Deep Spiking Neural Networks", P. O'Connor and Max Welling
    https://arxiv.org/pdf/1602.08323.pdf
    """
    phi = phi + v
    phi_abs = torch.abs(phi)
    phi_abs_sum = torch.sum(phi_abs)
    phi_abs_sorted, indices = torch.sort(phi_abs)
    phi_abs_summed = torch.cumsum(phi_abs_sorted, dim=1)
    z = torch.sign(phi) * torch.sign((phi_abs_sum - phi_abs_summed - 0.5))[indices]
    phi = phi - z
    return z, phi


def constant_spiking_vector_quantization(v: torch.Tensor, T: int):
    phi = torch.zeros_like(v)
    signed_spikes = []

    for _ in range(T):
        z, phi = spiking_vector_quantization_step(v, phi)
        signed_spikes.append(z)

    return torch.stack(signed_spikes)
