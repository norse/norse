import torch
from norse.torch.functional import receptive_field


def test_spatial_rf():
    rf = receptive_field.spatial_receptive_field(1, 1, 0, size=9)
    assert rf.shape == (9, 9)


def test_spatial_rf_backward():
    angle = torch.tensor([0.0], requires_grad=True)
    ratio = torch.tensor([1.0], requires_grad=True)
    scale = torch.tensor([5.0], requires_grad=True)
    rf = receptive_field.spatial_receptive_field(scale, angle, ratio, 9)
    assert rf.shape == (9, 9)
    assert rf.requires_grad

    rf.sum().backward()
    assert angle.grad is not None
    assert ratio.grad is not None
    assert scale.grad is not None


def test_spatial_parameters_derivative():
    scales = torch.tensor([0.2, 0.5, 1.0], requires_grad=True)
    angles = torch.tensor([0.0, 0.5 * torch.pi, torch.pi], requires_grad=True)
    ratios = torch.tensor([0.2, 0.5, 1.0], requires_grad=True)
    derivatives = 1
    sp = receptive_field.spatial_parameters(
        scales, angles, ratios, derivatives, include_replicas=True
    )
    assert sp.shape == (108, 5)
    sp.sum().backward()
    assert not sp.grad_fn is None

    assert scales.grad_fn is None
    assert angles.grad_fn is None
    assert ratios.grad_fn is None


def test_generate_fields():
    p = torch.tensor([[1, 1, 1, 1, 1.0]], requires_grad=True)
    f = receptive_field.spatial_receptive_fields_with_derivatives(p, 9)
    assert f.shape == (1, 9, 9)
    f.sum().backward()
    assert p.grad is not None
    assert p.grad_fn is None
