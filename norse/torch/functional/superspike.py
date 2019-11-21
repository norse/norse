import torch
from .heaviside import heaviside


class SuperSpike(torch.autograd.Function):
    """SuperSpike surrogate gradient as described in Section 3.3.2 of
    
        F. Zenke, S. Ganguli, "SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks", 
        Neural Computation 30, 1514–1541 (2018),
        doi:10.1162/neco_a_01086
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return heaviside(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        grad = grad_input / (alpha * torch.abs(input) + 1.0).pow(
            2
        )  # section 3.3.2 (beta -> alpha)
        return grad, None


def super_fn(x: torch.Tensor, alpha: float = 100.0) -> torch.Tensor:
    return SuperSpike.apply(x, alpha)
