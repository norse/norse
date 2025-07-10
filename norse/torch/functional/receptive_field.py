"""
A module for creating receptive fields.
"""

from typing import List, Tuple, Union, Optional

import torch


def gaussian_kernel(
    size: int, c: torch.Tensor, x: torch.Tensor, y: torch.Tensor, domain: int = 8
) -> torch.Tensor:
    """
    Efficiently creates a differentiable 2d gaussian kernel.

    Arguments:
      size (int): The size of the kernel
      c (torch.Tensor): A 2x2 covariance matrix describing the eccentricity of the gaussian
      x (torch.Tensor): The receptive's field center position in x-axis
      y (torch.Tensor): The receptive's field center position in y-axis
      domain (int): The domain of the kernel. Defaults to 8 (sampling -8 to 8).
    """
    ci = torch.linalg.inv(c)
    cd = torch.linalg.det(c)
    fraction = 1 / (2 * torch.pi * torch.sqrt(cd))
    a = torch.linspace(-domain, domain, size).to(c.device)
    xs, ys = torch.meshgrid(a, a, indexing="xy")
    xs = xs - x
    ys = ys - y
    coo = torch.stack([xs, ys], dim=2)
    b = torch.einsum("bimj,jk->bik", -coo.unsqueeze(2), ci)
    a = torch.einsum("bij,bij->bi", b, coo)
    return fraction * torch.exp(a / 2)


def covariance_matrix(
    sigma1: torch.Tensor, sigma2: torch.Tensor, phi: torch.Tensor
) -> torch.Tensor:
    """
    Creates a 2-dimensional covariance matrix given two variances and an angle for the major axis.
    """
    lambda1 = torch.as_tensor(sigma1) ** 2
    lambda2 = torch.as_tensor(sigma2) ** 2
    phi = torch.as_tensor(phi)
    cxx = lambda1 * phi.cos() ** 2 + lambda2 * phi.sin() ** 2
    cxy = (lambda1 - lambda2) * phi.cos() * phi.sin()
    cyy = lambda1 * phi.sin() ** 2 + lambda2 * phi.cos() ** 2
    cov = torch.ones(2, 2, device=phi.device)
    cov[0][0] = cxx
    cov[0][1] = cxy
    cov[1][0] = cxy
    cov[1][1] = cyy
    return cov


def derive_kernel(kernel, angle) -> torch.Tensor:
    """
    Takes the spatial derivative at a given angle
    """
    dirx = torch.cos(angle)
    diry = torch.sin(angle)
    grady, gradx = torch.gradient(kernel)
    return dirx * gradx + diry * grady


def calculate_normalization(dx: int, scale: float, gamma: float = 1):
    """
    Calculates scale normalization for a spatial receptive field at a given directional derivative
    Lindeberg: Feature detection with automatic scale selection, eq. 20
    https://doi.org/10.1023/A:1008045108935

    Arguments:
        dx (int): The nth directional derivative
        scale (float): The scale of the receptive field
        gamma (float): A normalization parameter
    """
    t = scale**2
    scale_norm = scale ** (dx * (1 - gamma))
    xi_norm = t ** (gamma / 2)
    return scale_norm * xi_norm


def derive_spatial_receptive_field_single(
    field: torch.Tensor, scale: float, angle: float, dx: int, dy: int
) -> torch.Tensor:
    """
    Calculate the derivative of a single spatial receptive field at a given angle and scale with respect to x and y derivatives.

    Example:
    >>> field = spatial_receptive_field(0, 1, 16)
    >>> derived = derive_spatial_receptive_field_xy(field, 0, 1, 1, 0)

    Arguments:
        field (torch.Tensor): The spatial receptive field
        scale (float): The scale of the receptive field
        angle (float): The angle of the receptive field
        dx (int): The x-th derivative
        dy (int): The y-th derivative

    Returns:
        torch.Tensor: The derived spatial receptive field
    """
    derived = field
    dx = int(dx)
    dy = int(dy)
    while dx > 0 or dy > 0:
        if dx > 0:
            derived = derive_kernel(derived, angle) * calculate_normalization(
                1, scale, 1
            )
            dx -= 1
        if dy > 0:
            derived = derive_kernel(
                derived, angle + torch.pi / 2
            ) * calculate_normalization(1, scale, 1)
            dy -= 1
    return derived


def derive_spatial_receptive_field(
    field: torch.Tensor, scale: float, angle: float, derivatives: List[Tuple[int, int]]
) -> torch.Tensor:
    """
    Derive spatial receptive field at a given angle and scale with respect to a list of derivatives.
    Returns a tensor of shape (len(derivatives), size, size), where size is the size of the receptive field.

    Arguments:
        field (torch.Tensor): The spatial receptive field
        scale (float): The scale of the receptive field
        angle (float): The angle of the receptive field
        derivatives (List[Tuple[int, int]]): A list of tuples of derivatives

    Returns:
        torch.Tensor: A list of derived spatial receptive field with the same length as the input list of derivatives
    """
    angle = torch.as_tensor(angle)
    kernels = []
    for dx, dy in derivatives:
        derived = derive_spatial_receptive_field_single(field, scale, angle, dx, dy)
        kernels.append(derived)
    return torch.stack(kernels)


def spatial_receptive_field(
    scale: torch.Tensor,
    angle: torch.Tensor,
    ratio: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    size: int,
    dx: int = 0,
    dy: int = 0,
    domain: float = 10,
) -> torch.Tensor:
    """
    Creates a (size x size) receptive field kernel at a given scale, angle and ratio with respect to x and y derivatives.

    Arguments:
      scale (torch.Tensor): The scale of the field. Defaults to 2.5
      angle (torch.Tensor): The rotation of the kernel in radians
      ratio (torch.Tensor): The eccentricity as a ratio
      x (torch.Tensor): The receptive's field center position in x-axis
      y (torch.Tensor): The receptive's field center position in y-axis
      size (int): The size of the square kernel in pixels
      dx (int): The x-th derivative of the field
      dy (int): The y-th derivative of the field
      domain (float): The initial coordinates from which the field is sampled. Defaults to 8 (sampling -8 to 8).
    """
    angle = torch.as_tensor(angle)
    c = covariance_matrix(ratio, 1 / ratio, angle) * scale
    k = gaussian_kernel(size, c, x, y, domain=domain)
    k = k / k.sum()
    return derive_spatial_receptive_field_single(k, scale, angle, dx, dy)


def _extract_derivatives(
    derivatives: Union[int, List[Tuple[int, int]]],
) -> Tuple[List[Tuple[int, int]], int]:
    if isinstance(derivatives, int):
        if derivatives == 0:
            return [(0, 0)], 0
        else:
            derivative_list = torch.arange(derivatives + 1)
            return (
                torch.cartesian_prod(derivative_list, derivative_list)[
                    :-1
                ].tolist(),  # Remove the final derivative (n, n) since it's higher-order than n
                derivatives,
            )
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
    x: torch.Tensor = torch.tensor([0.0]),
    y: torch.Tensor = torch.tensor([0.0]),
    include_replicas: bool = False,
) -> torch.Tensor:
    """
    Combines the parameters of scales, angles, ratios, xand y coordinates of the center of the rf and derivatives as cartesian products
    to produce a set of parameters for spatial receptive fields.

    Arguments:
        scales (torch.Tensor): The scales of the receptive fields
        angles (torch.Tensor): The angles of the receptive fields
        ratios (torch.Tensor): The ratios of the receptive fields
        derivatives (Union[int, List[Tuple[int, int]]): The derivatives of the receptive fields
        x (torch.Tensor): The x-shift of the receptive field center
        y (torch.Tensor): The y-shift of the receptive field center
        include_replicas (bool): If True, includes replicas of the receptive field for ratio 1 (they are symmetric under rotation). Defaults to False.
    """
    if include_replicas or not (ratios == 1).any():
        parameters = torch.cartesian_prod(scales, angles, ratios, x, y)
    else:
        mask = ratios != 1
        asymmetric_ratios = ratios[mask]
        symmetric_ratios = ratios[~mask]
        asymmetric_fields = torch.cartesian_prod(
            scales, angles, asymmetric_ratios, x, y
        )
        symmetric_rings = torch.cartesian_prod(
            scales,
            angles[:1],  # We can ignore angles when ratio is 1
            symmetric_ratios,
            x,
            y,
        )
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
    domain: float = 1,
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
                x=p[3],
                y=p[4],
                size=size,
                dx=p[5],
                dy=p[6],
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
    p = spatial_parameters(scales, angles, ratios, derivatives, include_replicas)
    return torch.cartesian_prod(p, temporal_scales)
