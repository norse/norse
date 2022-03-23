import torch
from pytest import raises


from norse.torch.functional.threshold import (
    SurrogateMethod,
    threshold,
    sign,
    heavi_erfc_fn,
    logistic_fn,
    circ_dist_fn,
)


def test_heavi_erfc_fn_forward():
    assert torch.equal(heavi_erfc_fn(torch.ones(100), 100.0), torch.ones(100))
    assert torch.equal(heavi_erfc_fn(-1.0 * torch.ones(100), 100.0), torch.zeros(100))


def test_heavi_erfc_fn_backward():
    x = torch.ones(10, requires_grad=True)
    out = heavi_erfc_fn(x, 0.1)
    out.backward(torch.ones(10))

    assert torch.sum(x.grad > 0) == 10

    x = torch.ones(10, requires_grad=True)
    out = heavi_erfc_fn(-0.001 * x, 0.1)
    out.backward(torch.ones(10))

    assert torch.sum(x.grad < 0) == 10


def test_logistic_fn_forward():
    x = torch.ones(10)
    out = logistic_fn(x, 0.1)
    assert torch.sum(torch.logical_or(out == 1, out == 0)) == 10


def test_logistic_fn_backward():
    x = torch.ones(10, requires_grad=True)
    out = logistic_fn(x, 0.1)
    out.backward(torch.ones(10))

    assert torch.sum(x.grad > 0) == 10

    x = torch.ones(10, requires_grad=True)
    out = logistic_fn(-0.001 * x, 0.1)
    out.backward(torch.ones(10))

    assert torch.sum(x.grad < 0) == 10


def test_circ_dist_fn_forward():
    x = torch.ones(10)
    out = circ_dist_fn(x, 0.1)
    assert torch.sum(torch.logical_or(out == 1, out == 0)) == 10


def test_circ_dist_fn_backward():
    x = torch.ones(10, requires_grad=True)
    out = circ_dist_fn(x, 0.1)
    out.backward(torch.ones(10))

    assert torch.sum(x.grad > 0) == 10

    x = torch.ones(10, requires_grad=True)
    out = circ_dist_fn(-0.001 * x, 0.1)
    out.backward(torch.ones(10))

    assert torch.sum(x.grad < 0) == 10


def test_threshold_backward():
    alpha = 10.0
    x = torch.ones(10)

    for method in SurrogateMethod:
        if method == SurrogateMethod.Heaviside:
            continue

        x = torch.ones(10, requires_grad=True)
        out = threshold(x, method, alpha)
        out.backward(torch.ones(10))

        x = torch.full((10,), 0.1, requires_grad=True)
        out = threshold(x, method, alpha)
        out.backward(torch.ones(10))

        x = torch.full((10,), -0.1, requires_grad=True)
        out = threshold(x, method, alpha)
        out.backward(torch.ones(10))


def test_threshold():
    alpha = 10.0

    for method in SurrogateMethod:
        x = torch.ones(10)
        out = threshold(x, method, alpha)
        assert torch.equal(out, torch.ones(10))

        x = torch.full((10,), 0.1)
        out = threshold(x, method, alpha)
        assert torch.equal(out, torch.ones(10))

        x = torch.full((10,), -0.1)
        out = threshold(x, method, alpha)
        assert torch.equal(out, torch.zeros(10))


def test_sign():
    alpha = 10.0

    for method in SurrogateMethod:
        x = torch.ones(10)
        out = sign(x, method, alpha)
        assert torch.equal(out, torch.ones(10))

        x = torch.full((10,), 0.1)
        out = sign(x, method, alpha)
        assert torch.equal(out, torch.ones(10))

        x = torch.full((10,), -0.1)
        out = sign(x, method, alpha)
        assert torch.equal(out, -torch.ones(10))
