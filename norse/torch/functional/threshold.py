import torch
import torch.jit
import numpy as np

import norse
from .heaviside import heaviside

if getattr(norse, "IS_OPS_LOADED"):
    superspike_fn = torch.ops.norse_op.superfun
else:
    from .superspike import super_fn

    superspike_fn = super_fn


class HeaviErfc(torch.autograd.Function):
    r"""Approximation of the heaviside step function as

    .. math::
        h(x,k) = \frac{1}{2} + \frac{1}{2} \text{erfc}(k x)

    where erfc is the error function.
    """

    @staticmethod
    def forward(ctx, x, k):
        ctx.save_for_backward(x)
        ctx.k = k
        return heaviside(x)  # 0 + 0.5 * torch.erfc(k * x)

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        derfc = (2 * torch.exp(-ctx.k.pow(2) * x.pow(2))) / (
            torch.as_tensor(np.pi).sqrt()
        )
        return derfc * dy, None


heavi_erfc_fn = HeaviErfc.apply


class HeaviTanh(torch.autograd.Function):
    r"""Approximation of the heaviside step function as

    .. math::
        h(x,k) = \frac{1}{2} + \frac{1}{2} \text{tanh}(k x)
    """

    @staticmethod
    def forward(ctx, x, k):
        ctx.save_for_backward(x)
        ctx.k = k
        return heaviside(x)  # 0.5 + 0.5 * torch.tanh(k * x)

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dtanh = 1 - (x * ctx.k).tanh().pow(2)
        return dy * dtanh, None


@torch.jit.ignore
def heavi_tanh_fn(x: torch.Tensor, k: float):
    return HeaviTanh.apply(x, k)


class Logistic(torch.autograd.Function):
    r"""Probalistic approximation of the heaviside step function as

    .. math::
        z \sim p(\frac{1}{2} + \frac{1}{2} \text{tanh}(k x))
    """

    @staticmethod
    def forward(ctx, x, k):
        ctx.k = k
        ctx.save_for_backward(x)
        p = 0.5 + 0.5 * torch.tanh(k * x)
        return torch.distributions.bernoulli.Bernoulli(probs=p).sample()

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dtanh = 1 - (x * ctx.k).tanh().pow(2)
        return dy * dtanh, None


@torch.jit.ignore
def logistic_fn(x: torch.Tensor, k: float):
    return Logistic.apply(x, k)


class HeaviCirc(torch.autograd.Function):
    r"""Approximation of the heaviside step function as

    .. math::
        h(x,\alpha) = \frac{1}{2} + \frac{1}{2} \
        \frac{x}{(x^2 + \alpha^2)^{1/2}}
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return heaviside(x)  # 0.5 + 0.5 * (x / (x.pow(2) + alpha ** 2).sqrt())

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha

        return (
            dy
            * (
                -(x.pow(2) / (2 * (alpha ** 2 + x.pow(2)).pow(1.5)))
                + 1 / (2 * (alpha ** 2 + x.pow(2)).sqrt())
            )
            * 2
            * alpha,
            None,
        )


@torch.jit.ignore
def heavi_circ_fn(x: torch.Tensor, k: float):
    return HeaviCirc.apply(x, k)


class CircDist(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return torch.distributions.bernoulli.Bernoulli(
            0.5 + 0.5 * (x / (x.pow(2) + alpha ** 2).sqrt())
        ).sample()

    @staticmethod
    def backward(ctx, dy):
        (
            x,
            alpha,
        ) = ctx.saved_tensors
        return (
            dy
            * (
                -(x.pow(2) / (2 * (alpha ** 2 + x.pow(2)).pow(1.5)))
                + 1 / (2 * (alpha ** 2 + x.pow(2)).sqrt())
            )
            * 2
            * alpha,
            None,
        )


@torch.jit.ignore
def circ_dist_fn(x: torch.Tensor, k: float):
    return CircDist.apply(x, k)


class HeaviTent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        ctx.save_for_backward(x)
        return heaviside(x)

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        return torch.relu(1 - torch.abs(x)) * alpha * dy, None


@torch.jit.ignore
def heavi_tent_fn(x: torch.Tensor, k: float):
    return HeaviTent.apply(x, k)


def threshold(x: torch.Tensor, method: str, alpha: float) -> torch.Tensor:
    if method == "heaviside":
        return heaviside(x)
    elif method == "super":
        return superspike_fn(x, torch.as_tensor(alpha))
    elif method == "tanh":
        return heavi_tanh_fn(x, alpha)
    elif method == "tent":
        return heavi_tent_fn(x, alpha)
    elif method == "circ":
        return heavi_circ_fn(x, alpha)
    elif method == "logistic":
        return logistic_fn(x, alpha)
    else:
        raise ValueError(
            f"Attempted to apply threshold function {method}, but no such "
            + "function exist. We currently support heaviside, super, "
            + "tanh, tent, circ, and logistic."
        )


def sign(x: torch.Tensor, method: str, alpha: float) -> torch.Tensor:
    return 2 * threshold(x, method, alpha) - 1
