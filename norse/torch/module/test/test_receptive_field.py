import pytest

import torch

from norse.torch import (
    SpatialReceptiveField2d,
    SampledSpatialReceptiveField2d,
    ParameterizedSpatialReceptiveField2d,
    TemporalReceptiveField,
)
from norse.torch.functional.receptive_field import spatial_parameters


def test_spatial_receptive_field():
    parameters = torch.tensor([[1, 1, 1, 0, 0, 0, 0.0]])
    m = SpatialReceptiveField2d(1, 9, parameters)
    assert m.weights.shape == (1, 1, 9, 9)


def test_spatial_receptive_field_update_flag():
    parameters = torch.tensor([[1, 1, 1, 0, 0, 0, 0.0]])
    m = SpatialReceptiveField2d(1, 9, parameters)
    assert not m.has_updated
    y = m(torch.empty(1, 1, 9, 9))
    y.sum().backward()
    assert m.has_updated


def test_spatial_receptive_field_update_kernels():
    parameters = torch.tensor([[1, 0, 1, 0, 0, 0, 0.0]])
    m = SpatialReceptiveField2d(1, 9, rf_parameters=parameters)
    old_kernels = m.weights.detach().clone()
    optim = torch.optim.SGD(list(m.parameters()), lr=1)
    y = m(torch.eye(9).view(1, 1, 9, 9))
    y.sum().backward()
    assert m.weights.grad_fn is not None
    assert m.rf_parameters.grad is not None and not torch.all(
        torch.eq(m.rf_parameters.grad, 0)
    )
    optim.step()
    m._update_weights()
    assert torch.all(torch.eq(m.rf_parameters, parameters))
    assert not torch.all(torch.eq(old_kernels, m.weights))


def test_spatially_parameterized_receptive_field_update():
    scales = torch.tensor([1.0, 2.0])
    angles = torch.tensor([0.0, 1.0])
    ratios = torch.tensor([0.5, 2.0])
    x = torch.tensor([0.0, 1.0])
    y = torch.tensor([0.0, 1.0])
    ratios_copy = ratios.clone()
    x_copy = x.clone()
    m = SampledSpatialReceptiveField2d(
        1, 9, scales, angles, ratios, 1, x, y, False, False, True, True, False
    )
    old_kernels = m.submodule.weights.detach().clone()
    optim = torch.optim.SGD(list(m.parameters()), lr=1)
    y = m(torch.ones(1, 1, 9, 9))
    y.sum().backward()
    assert m.submodule.weights.grad_fn is not None
    assert m.ratios.grad is not None
    assert m.scales.grad is None
    assert m.angles.grad is None
    assert m.x.grad is not None
    assert m.y.grad is None
    assert m.has_updated is True
    optim.step()
    m._update_weights()
    assert not torch.all(torch.eq(ratios, ratios_copy))
    assert not torch.all(torch.eq(x, x_copy))
    assert not torch.all(torch.eq(old_kernels, m.submodule.weights))


def test_column_parameterized_receptive_field_update():
    scales = torch.tensor([1.0, 2.0])
    angles = torch.tensor([0.0, 1.0])
    ratios = torch.tensor([0.5, 1.0])
    x = torch.tensor([0.0, 1.0])
    y = torch.tensor([0.0, 1.0])
    m = ParameterizedSpatialReceptiveField2d(
        1, 9, scales, angles, ratios, 1, x, y, False, False, True, True, False
    )
    ratios_copy = m.ratios.clone()
    x_copy = m.x.clone()
    old_kernels = m.submodule.weights.detach().clone()
    optim = torch.optim.SGD(list(m.parameters()), lr=1)
    y = m(torch.ones(1, 1, 9, 9))
    y.sum().backward()
    assert m.submodule.weights.grad_fn is not None
    assert m.ratios.grad is not None
    assert m.scales.grad is None
    assert m.angles.grad is None
    assert m.x.grad is not None
    assert m.y.grad is None
    assert m.has_updated is True
    optim.step()
    print(m.parameters())
    m._update_weights()
    assert not torch.all(torch.eq(m.ratios, ratios_copy))
    assert not torch.all(torch.eq(m.x, x_copy))
    assert not torch.all(torch.eq(old_kernels, m.submodule.weights))


def test_column_parameterized_receptive_field_update_default_xy():
    scales = torch.tensor([1.0, 2.0])
    angles = torch.tensor([0.0, 1.0])
    ratios = torch.tensor([0.5, 1.0])
    x = torch.tensor([0.0, 1.0])
    y = torch.tensor([0.0, 1.0])
    m = ParameterizedSpatialReceptiveField2d(
        1,
        9,
        scales,
        angles,
        ratios,
        1,
        optimize_scales=False,
        optimize_angles=False,
        optimize_ratios=True,
        optimize_x=True,
        optimize_y=False,
    )
    ratios_copy = m.ratios.clone()
    x_copy = m.x.clone()
    old_kernels = m.submodule.weights.detach().clone()
    optim = torch.optim.SGD(list(m.parameters()), lr=1)
    y = m(torch.ones(1, 1, 9, 9))
    y.sum().backward()
    assert m.submodule.weights.grad_fn is not None
    assert m.ratios.grad is not None
    assert m.scales.grad is None
    assert m.angles.grad is None
    assert m.x.grad is not None
    assert m.y.grad is None
    assert m.has_updated is True
    optim.step()
    print(m.parameters())
    m._update_weights()
    assert not torch.all(torch.eq(m.ratios, ratios_copy))
    assert not torch.all(torch.eq(m.x, x_copy))
    assert not torch.all(torch.eq(old_kernels, m.submodule.weights))


def test_backprop_twice():
    model = SpatialReceptiveField2d(1, 9, torch.tensor([[1, 1, 1, 0, 0, 0, 0.0]]))
    x = torch.ones(1, 1, 9, 9)
    y1 = model(x)
    y1.sum().backward()
    y2 = model(x)
    y2.sum().backward()

    # This should fail
    with pytest.raises(RuntimeError):
        y1 = model(x)
        y1.sum().backward()
        model.has_updated = False
        y2 = model(x)
        y2.sum().backward()


def test_backprop_twice_log_upd():
    model = SpatialReceptiveField2d(
        1, 9, torch.tensor([[1, 1, 1, 0, 0, 0, 0.0]]), optimize_log=True
    )
    x = torch.ones(1, 1, 9, 9)
    y1 = model(x)
    y1.sum().backward()
    y2 = model(x)
    y2.sum().backward()

    # This should fail
    with pytest.raises(RuntimeError):
        y1 = model(x)
        y1.sum().backward()
        model.has_updated = False
        y2 = model(x)
        y2.sum().backward()


def test_column_parameterized_receptive_field_logarithmic_update():
    scales = torch.tensor([1.0, 2.0])
    angles = torch.tensor([0.0, 1.0])
    ratios = torch.tensor([0.5, 1.0])
    x = torch.tensor([0.0, 1.0])
    y = torch.tensor([0.0, 1.0])
    m = ParameterizedSpatialReceptiveField2d(
        1, 9, scales, angles, ratios, 1, x, y, False, False, True, True, False, True
    )
    ratios_copy = m.ratios.clone()
    x_copy = m.x.clone()
    old_kernels = m.submodule.weights.detach().clone()
    optim = torch.optim.SGD(list(m.parameters()), lr=1)
    y = m(torch.ones(1, 1, 9, 9))
    y.sum().backward()
    assert m.submodule.weights.grad_fn is not None
    assert not m.ratios.is_leaf
    assert m.log_ratios.grad is not None
    assert m.scales.grad is None
    assert m.angles.grad is None
    assert m.x.grad is not None
    assert m.y.grad is None
    assert m.has_updated is True
    optim.step()
    print(m.parameters())
    m._update_weights()
    assert not torch.all(torch.eq(m.ratios, ratios_copy))
    assert not torch.all(torch.eq(m.log_ratios, torch.log(ratios_copy)))
    assert not torch.all(torch.eq(m.x, x_copy))
    assert not torch.all(torch.eq(old_kernels, m.submodule.weights))
