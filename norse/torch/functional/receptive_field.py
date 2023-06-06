"""
These receptive fields are derived from scale-space theory, specifically in the paper `Normative theory of visual receptive fields by Lindeberg, 2021 <https://www.sciencedirect.com/science/article/pii/S2405844021000025>`_.

For use in spiking / binary signals, see the paper on `Translation and Scale Invariance for Event-Based Object tracking by Pedersen et al., 2023 <https://dl.acm.org/doi/10.1145/3584954.3584996>`_
"""

from typing import List, Tuple, Union

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


def spatial_receptive_field(
    angle, ratio, size: int, scale: float = 2.5, domain: float = 8
):
    """
    Creates a (size x size) receptive field kernel

    Arguments:
      angle (float): The rotation of the kernel in radians
      ratio (float): The eccentricity as a ratio
      size (int): The size of the square kernel in pixels
      scale (float): The scale of the field. Defaults to 2.5
      domain (float): The initial coordinates from which the field is sampled. Defaults to 8 (equal to -8 to 8).
    """
    sm = torch.tensor([scale, scale * ratio])
    a = torch.linspace(-domain, domain, size)
    r = torch.tensor(
        [[torch.cos(angle), torch.sin(angle)], [-torch.sin(angle), torch.cos(angle)]],
        dtype=torch.float32,
    )
    c = (r * sm) @ (sm * r).T
    xs, ys = torch.meshgrid(a, a, indexing="xy")
    coo = torch.stack([xs, ys], dim=2)
    return gaussian_kernel(coo, scale, c)


def _extract_derivatives(
    derivatives: Union[int, List[Tuple[int, int]]]
) -> Tuple[List[Tuple[int, int]], int]:
    if isinstance(derivatives, int):
        if derivatives == 0:
            return [(0, 0)], 0
        else:
            return [
                (x, y) for x in range(derivatives + 1) for y in range(derivatives + 1)
            ], derivatives
    elif isinstance(derivatives, list):
        return derivatives, max([max(x, y) for (x, y) in derivatives])
    else:
        raise ValueError(
            f"Derivatives expected either a number or a list of tuples, but got {derivatives}"
        )


def _derive_fields(
    fields: torch.Tensor, derivatives: List[Tuple[int, int]]
) -> torch.Tensor:
    out = []
    for dx, dy in derivatives:
        if dx == 0:
            fx = fields
        else:
            fx = fields.diff(
                dim=1, prepend=torch.zeros(fields.shape[0], dx, fields.shape[2]), n=dx
            )

        if dy == 0:
            fy = fx
        else:
            fy = fx.diff(
                dim=2, prepend=torch.zeros(fields.shape[0], fields.shape[1], dy), n=dy
            )
        out.append(fy)
    return torch.concat(out)


def spatial_receptive_fields_with_derivatives(
    n_scales: int,
    n_angles: int,
    n_ratios: int,
    size: int,
    derivatives: Union[int, List[Tuple[int, int]]] = 0,
) -> torch.Tensor:
    r"""
    Creates a number of receptive field with 1st directional derivatives.
    The parameters decide the number of combinations to scan over, i. e. the number of receptive fields to generate.
    Specifically, we generate ``derivatives * (n_angles * n_scales * (n_ratios - 1) + n_scales)`` fields.

    The ``(n_ratios - 1) + n_scales`` terms exist because at ``ratio = 1``, fields are perfectly symmetrical, and there
    is therefore no reason to scan over the angles and scales for ``ratio = 1``.
    However, ``n_scales`` receptive fields still need to be added (one for each scale-space).
    Finally, the ``derivatives *`` term comes from the addition of spatial derivatives.

    Arguments:
        n_scales (int): Number of scaling combinations (the size of the receptive field) drawn from a logarithmic distribution
        n_angles (int): Number of angular combinations (the orientation of the receptive field)
        n_ratios (int): Number of eccentricity combinations (how "flat" the receptive field is)
        size (int): The size of the square kernel in pixels
        derivatives (Union[int, List[Tuple[int, int]]]): The spatial derivatives to include. Defaults to 0 (no derivatives).
            Can either be a number, in which case 1 + 2 ** n derivatives will be made (except when 0, see below).
              Example: `derivatives=0` omits derivatives
              Example: `derivatives=1` provides 2 spatial derivatives + 1 without derivation
            Or a list of tuples specifying the derivatives in both spatial dimensions
              Example: `derivatives=[(0, 0), (1, 2)]` provides two outputs, one without derivation and one :math:`\partial_x \partial^2_y`
    """
    angles = torch.linspace(0, torch.pi - torch.pi / n_angles, n_angles)
    ratios = torch.linspace(0.25, 1, n_ratios)
    scales = torch.exp(torch.linspace(0.5, 1.5, n_scales))
    # We add extra space in both the domain and size to account for the derivatives
    derivative_list, derivative_max = _extract_derivatives(derivatives)
    domain = 8 + derivative_max * size * 0.5

    assymmetric_rings = torch.stack(
        [
            spatial_receptive_field(
                angle, ratio, size=size + 2 * derivative_max, scale=scale, domain=domain
            )
            for angle in angles
            for scale in scales
            for ratio in ratios[:-1]
        ]
    )
    symmetric_rings = torch.stack(
        [
            spatial_receptive_field(
                torch.as_tensor(0),
                torch.as_tensor(1),
                size + 2 * derivative_max,
                scale=scale,
                domain=domain,
            )
            for scale in scales
        ]
    )
    rings = torch.concat([assymmetric_rings, symmetric_rings])
    derived_fields = _derive_fields(rings, derivative_list)
    return derived_fields[
        :,
        derivative_max : size + derivative_max,
        derivative_max : size + derivative_max,  # Remove extra space
    ]


def temporal_scale_distribution(start: float, stop: float, steps: int):
    return torch.exp(
        torch.linspace(
            torch.log(torch.as_tensor(start)), torch.log(torch.as_tensor(stop)), steps
        )
    )
