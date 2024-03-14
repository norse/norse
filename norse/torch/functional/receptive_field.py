from typing import List, Tuple, Union, Optional

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
    scale: float,
    angle: float,
    ratio: float,
    size: int,
    dx: int = 0,
    dy: int = 0,
    domain: float = 8,
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
    sm = torch.ones(2)
    sm[0] = scale
    sm[1] = scale * ratio
    a = torch.linspace(-domain, domain, size)
    angle = torch.as_tensor(angle)
    r = torch.ones((2, 2), device=angle.device)
    r[0][0] = angle.cos()
    r[0][1] = angle.sin()
    r[1][0] = -angle.sin()
    r[1][1] = angle.cos()
    c = (r * sm) @ (sm * r).T
    xs, ys = torch.meshgrid(a, a, indexing="xy")
    coo = torch.stack([xs, ys], dim=2)
    k = gaussian_kernel(coo, scale, c)
    k = _derived_field(k, (dx, dy))
    return k


def _derived_field(field: torch.Tensor, derivatives: Tuple[int, int]) -> torch.Tensor:
    out = []
    (dx, dy) = derivatives
    device = field.device
    if dx == 0:
        fx = field
    else:
        fx = field.diff(
            dim=0,
            prepend=torch.zeros(int(dx.item()), field.shape[1]).to(device),
            n=int(dx.item()),
        )
    if dy == 0:
        fy = fx
    else:
        fy = fx.diff(
            dim=1,
            prepend=torch.zeros(field.shape[0], int(dy.item())).to(device),
            n=int(dy.item()),
        )
    out.append(fy)
    return torch.concat(out)


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


def spatial_parameters(
    scales: torch.Tensor,
    angles: torch.Tensor,
    ratios: torch.Tensor,
    derivatives: Union[int, List[Tuple[int, int]]],
    include_replicas: bool = False,
) -> torch.Tensor:
    """
    Combines the parameters of scales, angles, ratios and derivatives as cartesian products
    to produce a set of parameters for spatial receptive fields.
    """
    if include_replicas or not (ratios == 1).any():
        parameters = torch.cartesian_prod(scales, angles, ratios)
    else:
        mask = ratios != 1
        asymmetric_ratios = ratios[mask]
        symmetric_ratios = ratios[~mask]
        asymmetric_fields = torch.cartesian_prod(scales, angles, asymmetric_ratios)
        symmetric_rings = torch.cartesian_prod(scales, angles, symmetric_ratios)
        parameters = torch.cat([asymmetric_fields, symmetric_rings])
    # Add derivatives
    derivatives, _ = _extract_derivatives(derivatives)
    derivatives = torch.tensor(derivatives, device=scales.device).float()
    parameters_repeated = parameters.repeat_interleave(len(derivatives), 0)
    derivatives_repeated = derivatives.repeat(len(parameters), 1)
    return torch.cat([parameters_repeated, derivatives_repeated], 1)


def spatial_receptive_fields_with_derivatives(
    combinations: torch.Tensor,
    size: int,
    domain: float = 8,
) -> torch.Tensor:
    r"""
    Creates a number of receptive fields based on the spatial parameters and size of the receptive field.
    """
    return torch.stack(
        [
            spatial_receptive_field(
                scale=p[0],
                angle=p[1],
                ratio=p[2],
                size=size,
                dx=p[3],
                dy=p[4],
                domain=domain,
            )
            for p in combinations
        ]
    )


def temporal_scale_distribution(
    n_scales: int,
    min_scale: float = 1,
    max_scale: Optional[float] = None,
    c: Optional[float] = 1.41421,
):
    r"""
    Provides temporal scales according to [Lindeberg2016].
    The scales will be logarithmic by default, but can be changed by providing other values for c.

    .. math:
        \tau_k = c^{2(k - K)} \tau_{max}
        \mu_k = \sqrt(\tau_k - \tau_{k - 1})

    Arguments:
      n_scales (int): Number of scales to generate
      min_scale (float): The minimum scale
      max_scale (Optional[float]): The maximum scale. Defaults to None. If set, c is ignored.
      c (Optional[float]): The base from which to generate scale values. Should be a value
        between 1 to 2, exclusive. Defaults to sqrt(2). Ignored if max_scale is set.

    .. [Lindeberg2016] Lindeberg 2016, Time-Causal and Time-Recursive Spatio-Temporal
        Receptive Fields, https://link.springer.com/article/10.1007/s10851-015-0613-9.
    """
    xs = torch.linspace(1, n_scales, n_scales)
    if max_scale is not None:
        if n_scales > 1:  # Avoid division by zero when having a single scale
            c = (min_scale / max_scale) ** (1 / (2 * (n_scales - 1)))
        else:
            return torch.tensor([min_scale]).sqrt()
    else:
        max_scale = (c ** (2 * (n_scales - 1))) * min_scale
    taus = c ** (2 * (xs - n_scales)) * max_scale
    return taus.sqrt()


def spatio_temporal_parameters(
    scales: torch.Tensor,
    angles: torch.Tensor,
    ratios: torch.Tensor,
    derivatives: Union[int, List[Tuple[int, int]]],
    temporal_scales: torch.Tensor,
    include_replicas: bool = False,
) -> torch.Tensor:
    """
    Combines the parameters of scales, angles, ratios and derivatives as cartesian products
    to produce a set of parameters for spatial receptive fields.
    """
    spatial_parameters = spatial_parameters(
        scales, angles, ratios, derivatives, include_replicas
    )
    return torch.cartesian_prod(spatial_parameters, temporal_scales)
