import torch
from norse.torch.functional.heaviside import heaviside


class SuperSpike(torch.autograd.Function):
    r"""SuperSpike surrogate gradient as described in Section 3.3.2 of

    F. Zenke, S. Ganguli, **"SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks"**,
    Neural Computation 30, 1514–1541 (2018),
    `doi:10.1162/neco_a_01086 <https://www.mitpressjournals.org/doi/full/10.1162/neco_a_01086>`_
    """

    @staticmethod
    def forward(input_tensor: torch.Tensor, alpha: float) -> torch.Tensor:
        return heaviside(input_tensor)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input_tensor, alpha = inputs
        ctx.alpha = alpha
        ctx.save_for_backward(input_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        grad = None
        if ctx.needs_input_grad[0]:
            grad = grad_output / (torch.abs(inp) + 1.0).pow(
                2
            )  # section 3.3.2 (beta -> alpha)
        return grad, None


def super_fn(x: torch.Tensor, alpha: float = 100.0) -> torch.Tensor:
    return SuperSpike.apply(x, alpha)
