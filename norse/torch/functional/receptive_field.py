"""
These receptive fields are derived from scale-space theory, specifically in the paper `Normative theory of visual receptive fields by Lindeberg, 2021 <https://www.sciencedirect.com/science/article/pii/S2405844021000025>`_.

For use in spiking / binary signals, see the paper on `Translation and Scale Invariance for Event-Based Object tracking by Pedersen et al., 2023 <https://dl.acm.org/doi/10.1145/3584954.3584996>`_
"""

import torch


def gaussian_kernel(x, s, c):
    """
    Efficiently creates a 2d gaussian kernel.

    Arguments:
      x (torch.Tensor): A 2-d matrix
      s (float): The variance of the gaussian
      c (torch.Tensor): A 2x2 covariance matrix describing the eccentricity of the gaussian
    """
    ci = torch.linalg.inv(c)
    cd = torch.linalg.det(c)
    fraction = 1 / (2 * torch.pi * s * torch.sqrt(cd))
    b = torch.einsum("bimj,jk->bik", -x.unsqueeze(2), ci)
    a = torch.einsum("bij,bij->bi", b, x)
    return fraction * torch.exp(a / (2 * s))


def receptive_field(angle, ratio, size: int, scale: float = 2.5):
    """
    Creates a (size x size) receptive field kernel

    Arguments:
      angle (float): The rotation of the kernel in radians
      ratio (float): The eccentricity as a ratio
      size (int): The size of the square kernel in pixels
      scale (float): The scale of the field. Defaults to 2.5
    """
    sm = torch.tensor([scale, scale * ratio])

    a = torch.linspace(-8, 8, size)
    r = torch.tensor(
        [[torch.cos(angle), torch.sin(angle)], [-torch.sin(angle), torch.cos(angle)]],
        dtype=torch.float32,
    )
    c = (r * sm) @ (sm * r).T
    xs, ys = torch.meshgrid(a, a, indexing="xy")
    coo = torch.stack([xs, ys], dim=2)
    return gaussian_kernel(coo, scale, c)


def receptive_fields_with_derivatives(
    n_scales: int, n_angles: int, n_ratios: int, size: int
):
    """
    Creates a number of receptive field with 1st directional derivatives.
    The parameters decide the number of combinations to scan over, i. e. the number of receptive fields to generate.
    Specifically, we generate ``3 * (n_angles * n_scales * (n_ratios - 1) + n_scales)`` fields.

    The ``(n_ratios - 1) + n_scales`` terms exist because at ``ratio = 1``, fields are perfectly symmetrical, and there
    is therefore no reason to scan over the angles and scales for ``ratio = 1``.
    However, ``n_scales`` receptive field still needs to be added (one for each scale-space).
    Finally, the ``3 *`` term comes from the addition of two spatial derivatives (1 in x, 1 in y).

    Arguments:
      n_scales (int): Number of scaling combinations (the size of the receptive field) drawn from a logarithmic distribution
      n_angles (int): Number of angular combinations (the orientation of the receptive field)
      n_ratios (int): Number of eccentricity combinations (how "flat" the receptive field is)
      size (int): The size of the square kernel in pixels
    """
    angles = torch.linspace(0, torch.pi - torch.pi / n_angles, n_angles)
    ratios = torch.linspace(0.25, 1, n_ratios)
    scales = torch.exp(torch.linspace(0.5, 1.2, n_scales))
    assymmetric_rings = torch.stack(
        [
            receptive_field(angle, ratio, size=9, scale=scale)
            for angle in angles
            for scale in scales
            for ratio in ratios[:-1]
        ]
    )
    symmetric_rings = torch.stack(
        [
            receptive_field(torch.as_tensor(0), torch.as_tensor(1), size, scale=scale)
            for scale in scales
        ]
    )
    rings = torch.concat([assymmetric_rings, symmetric_rings])
    ringsx = rings.diff(
        dim=1,
        prepend=torch.zeros(n_angles * n_scales * (n_ratios - 1)
                             + n_scales, 1, 9),
    )
    ringsy = rings.diff(
        dim=2,
        prepend=torch.zeros(n_angles * n_scales * (n_ratios - 1) + n_scales, 9, 1),
    )
    return torch.concat([rings, ringsx, ringsy])
