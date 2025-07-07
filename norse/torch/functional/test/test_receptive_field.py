import platform
import pytest

import torch
from norse.torch.functional import receptive_field


def test_spatial_rf():
    rf = receptive_field.spatial_receptive_field(1, 0, 1, 0, 0, size=9)
    assert rf.shape == (9, 9)


@pytest.mark.parametrize(
    "n_scales, n_angles, n_ratios, n_derivatives",
    [
        (a, b, c, d)
        for a in range(1, 4)
        for b in range(1, 4)
        for c in range(1, 4)
        for d in [[(0, 0)], [(0, 0), (1, 0)]]
    ],
)
def test_spatial_parameters_redundant_angles(
    n_scales, n_angles, n_ratios, n_derivatives
):
    scales = (2 ** torch.arange(n_scales)).float()
    angles = torch.linspace(0, torch.pi - torch.pi / n_angles, n_angles).float()
    ratios = (2 ** torch.arange(n_ratios)).float()
    params = receptive_field.spatial_parameters(
        scales=scales,
        angles=angles,
        ratios=ratios,
        derivatives=n_derivatives,
        include_replicas=False,
    )
    assert params.shape == (
        n_scales * (n_angles - 1) * (n_ratios - 1) * len(n_derivatives)
        + n_scales * n_ratios * len(n_derivatives),
        7,
    )

    params = receptive_field.spatial_parameters(
        scales=scales,
        angles=angles,
        ratios=ratios,
        derivatives=n_derivatives,
        include_replicas=True,
    )
    assert params.shape == (n_scales * n_angles * n_ratios * len(n_derivatives), 7)


def test_gaussian_kernel_backwards():
    angle = torch.tensor([1.0], requires_grad=True)
    ratio = torch.tensor([2.0], requires_grad=True)
    x = torch.tensor([1.0], requires_grad=True)
    y = torch.tensor([1.0], requires_grad=True)
    c = receptive_field.covariance_matrix(ratio, 1 / ratio, angle)
    kernel = receptive_field.gaussian_kernel(9, c, x, y)
    assert kernel.shape == (9, 9)
    assert kernel.requires_grad

    kernel.sum().backward()
    assert angle.grad is not None and angle.grad != 0
    assert ratio.grad is not None and ratio.grad != 0
    assert x.grad is not None and x.grad != 0
    assert y.grad is not None and y.grad != 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")
def test_gaussian_kernel_backwards_cuda():
    angle = torch.tensor([1.0], requires_grad=True, device="cuda")
    ratio = torch.tensor([2.0], requires_grad=True, device="cuda")
    x = torch.tensor([1.0], requires_grad=True, device="cuda")
    y = torch.tensor([1.0], requires_grad=True, device="cuda")
    c = receptive_field.covariance_matrix(ratio, 1 / ratio, angle)
    kernel = receptive_field.gaussian_kernel(9, c, x, y)
    assert kernel.shape == (9, 9)
    assert kernel.requires_grad

    kernel.sum().backward()
    assert angle.grad is not None and angle.grad != 0
    assert angle.grad.device.type == "cuda"
    assert ratio.grad is not None and ratio.grad != 0
    assert ratio.grad.device.type == "cuda"
    assert x.grad is not None and x.grad != 0
    assert x.grad.device.type == "cuda"
    assert y.grad is not None and y.grad != 0
    assert y.grad.device.type == "cuda"


@pytest.mark.skipif(platform.system() == "Darwin", reason="Mac test failing")
def test_spatial_rf_backward():
    angle = torch.tensor([0.0], requires_grad=True)
    ratio = torch.tensor([2.0], requires_grad=True)
    scale = torch.tensor([1.0], requires_grad=True)
    x = torch.tensor([0.0], requires_grad=True)
    y = torch.tensor([0.0], requires_grad=True)

    rf = receptive_field.spatial_receptive_field(scale, angle, ratio, x, y, 9)
    assert rf.shape == (9, 9)
    assert rf.requires_grad

    rf.sum().backward()
    assert angle.grad is not None and angle.grad != 0
    assert ratio.grad is not None and ratio.grad != 0
    assert scale.grad is not None and scale.grad != 0
    assert x.grad is not None and x.grad != 0
    assert y.grad is not None and y.grad != 0


def test_covariance_backward():
    ratio = torch.tensor([2.0], requires_grad=True)
    angle = torch.tensor([3.0], requires_grad=True)
    c = receptive_field.covariance_matrix(ratio, 1 / ratio, angle)
    c.sum().backward()
    assert ratio.grad is not None and ratio.grad != 0
    assert angle.grad is not None and angle.grad != 0


def test_derive_backward():
    field = torch.randn(9, 9, requires_grad=True)
    angle = torch.tensor([0.0], requires_grad=True)
    out = receptive_field.derive_kernel(field, angle)
    out.sum().backward()
    assert angle.grad is not None and angle.grad != 0
    assert field.grad is not None and not torch.all(torch.eq(field.grad, 0))


def test_spatial_parameters_derivative():
    scales = torch.tensor([0.2, 0.5, 1.0], requires_grad=True)
    angles = torch.tensor([0.0, 0.5 * torch.pi, torch.pi], requires_grad=True)
    ratios = torch.tensor([0.2, 0.5, 1.0], requires_grad=True)
    x = torch.tensor([0.0, 0.0], requires_grad=True)
    y = torch.tensor([0.0, 0.0], requires_grad=True)

    derivatives = 1
    sp = receptive_field.spatial_parameters(
        scales, angles, ratios, derivatives, x, y, include_replicas=True
    )
    assert sp.shape == (324, 7)
    sp = receptive_field.spatial_parameters(
        scales,
        angles,
        ratios,
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        x,
        y,
        include_replicas=True,
    )
    assert sp.shape == (432, 7)
    sp.sum().backward()
    assert not sp.grad_fn is None

    assert scales.grad_fn is None
    assert angles.grad_fn is None
    assert ratios.grad_fn is None
    assert x.grad_fn is None
    assert y.grad_fn is None


def test_generate_fields():
    p = torch.tensor([[1, 1, 1, 1, 1, 1, 1.0]], requires_grad=True)
    f = receptive_field.spatial_receptive_fields_with_derivatives(p, 9)
    assert f.shape == (1, 9, 9)
    f.sum().backward()
    assert p.grad is not None
    assert p.grad_fn is None
