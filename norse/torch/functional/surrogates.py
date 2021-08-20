from typing import Optional

import numpy as np
import torch


__all__ = ["erfc", "tanh", "logistic", "circ", "superspike", "triangle"]


def erfc(x: torch.Tensor, alpha: float, beta: Optional[float]) -> torch.Tensor:
    r"""Approximation of the heaviside step function as

    .. math::
        h(x,\alpha) = \frac{1}{2} + \frac{1}{2} \text{erfc}(\alpha x)

    where erfc is the error function.
    """
    return (2 * torch.exp(-(alpha * x).pow(2))) / (torch.as_tensor(np.pi).sqrt())


def tanh(x: torch.Tensor, alpha: float, beta: Optional[float]) -> torch.Tensor:
    r"""Approximation of the heaviside step function as

    .. math::
        h(x,\alpha) = \frac{1}{2} + \frac{1}{2} \text{tanh}(\alpha x)
    """
    return 1 - (x * alpha).tanh().pow(2)


def logistic(x: torch.Tensor, alpha: float, beta: Optional[float]) -> torch.Tensor:
    r"""Probalistic approximation of the heaviside step function as

    .. math::
        z \sim p(\frac{1}{2} + \frac{1}{2} \text{tanh}(\alpha x))
    """
    return 1 - (x * alpha).tanh().pow(2)


def circ(x: torch.Tensor, alpha: float, beta: Optional[float]) -> torch.Tensor:
    r"""Approximation of the heaviside step function as

    .. math::
        h(x,\alpha) = \frac{1}{2} + \frac{1}{2} \
        \frac{x}{(x^2 + \alpha^2)^{1/2}}
    """

    return -(x.pow(2) / (2 * (alpha ** 2 + x.pow(2)).pow(1.5))) + 1 / (2 * (alpha ** 2 + x.pow(2)).sqrt()) * 2 * alpha


def superspike(x: torch.Tensor, alpha: float, beta: Optional[float]) -> torch.Tensor:
    r"""SuperSpike surrogate gradient as described in Section 3.3.2 of

    F. Zenke, S. Ganguli, **"SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks"**,
    Neural Computation 30, 1514–1541 (2018),
    `doi:10.1162/neco_a_01086 <https://www.mitpressjournals.org/doi/full/10.1162/neco_a_01086>`_
    """
    return 1 / (alpha * torch.abs(x) + 1).pow(2)


def triangle(x: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    r"""Triangular/piecewise linear surrogate gradient as in

    S.K. Esser et al., **"Convolutional networks for fast, energy-efficient neuromorphic computing"**,
    Proceedings of the National Academy of Sciences 113(41), 11441-11446, (2016),
    `doi:10.1073/pnas.1604850113 <https://www.pnas.org/content/113/41/11441.short>`_
    G. Bellec et al., **"A solution to the learning dilemma for recurrent networks of spiking neurons"**,
    Nature Communications 11(1), 3625, (2020),
    `doi:10.1038/s41467-020-17236-y <https://www.nature.com/articles/s41467-020-17236-y>`_
    """
    return beta * torch.relu(1 - alpha * x.abs())
