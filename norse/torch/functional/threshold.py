import torch
import torch.jit
import numpy as np

from norse.torch.functional.heaviside import heaviside
from norse.torch.functional.superspike import super_fn

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
        derfc = (2 * torch.exp(-(ctx.k * x).pow(2))) / (torch.as_tensor(np.pi).sqrt())
        return derfc * dy, None


@torch.jit.ignore
def heavi_erfc_fn(x: torch.Tensor, k: float):
    return HeaviErfc.apply(x, k)


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
                -(x.pow(2) / (2 * (alpha**2 + x.pow(2)).pow(1.5)))
                + 1 / (2 * (alpha**2 + x.pow(2)).sqrt())
            )
            * 2
            * alpha,
            None,
        )


@torch.jit.ignore
def heavi_circ_fn(x: torch.Tensor, k: float):
    return HeaviCirc.apply(x, k)


class CircDist(torch.autograd.Function):
    r"""Approximation of the heaviside step function as

    .. math::
        h(x,\alpha) = 0.5 + 0.5 * \frac{x}{\sqrt{x^2 + alpha^2}}
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha

        return torch.distributions.bernoulli.Bernoulli(
            0.5 + 0.5 * (x / (x.pow(2) + alpha**2).sqrt())
        ).sample()

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        return (
            dy
            * (
                -(x.pow(2) / (2 * (alpha**2 + x.pow(2)).pow(1.5)))
                + 1 / (2 * (alpha**2 + x.pow(2)).sqrt())
            )
            * 2
            * alpha,
            None,
        )


@torch.jit.ignore
def circ_dist_fn(x: torch.Tensor, k: float):
    return CircDist.apply(x, k)


class Triangle(torch.autograd.Function):
    r"""Triangular/piecewise linear surrogate gradient as in

    S.K. Esser et al., **"Convolutional networks for fast, energy-efficient neuromorphic computing"**,
    Proceedings of the National Academy of Sciences 113(41), 11441-11446, (2016),
    `doi:10.1073/pnas.1604850113 <https://www.pnas.org/content/113/41/11441.short>`_
    G. Bellec et al., **"A solution to the learning dilemma for recurrent networks of spiking neurons"**,
    Nature Communications 11(1), 3625, (2020),
    `doi:10.1038/s41467-020-17236-y <https://www.nature.com/articles/s41467-020-17236-y>`_
    """

    @staticmethod
    @torch.jit.ignore
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    @torch.jit.ignore
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        grad = grad_input * alpha * torch.relu(1 - x.abs())
        return grad, None


@torch.jit.ignore
def triangle_fn(x: torch.Tensor, alpha: float = 0.3) -> torch.Tensor:
    return Triangle.apply(x, alpha)


def threshold(x: torch.Tensor, method: str, alpha: float) -> torch.Tensor:
    if method == "heaviside":
        return heaviside(x)
    elif method == "super":
        return superspike_fn(x, torch.as_tensor(alpha))
    elif method == "triangle":
        return triangle_fn(x, alpha)
    elif method == "tanh":
        return heavi_tanh_fn(x, alpha)
    elif method == "circ":
        return heavi_circ_fn(x, alpha)
    elif method == "heavi_erfc":
        return heavi_erfc_fn(x, alpha)
    else:
        raise ValueError(
            f"Attempted to apply threshold function {method}, but no such "
            + "function exist. We currently support heaviside, super, "
            + "tanh, triangle, circ, and heavi_erfc."
        )


def sign(x: torch.Tensor, method: str, alpha: float) -> torch.Tensor:
    return 2 * threshold(x, method, alpha) - 1
