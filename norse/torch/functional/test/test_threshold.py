import torch
from pytest import raises


from norse.torch.functional.threshold import threshold, sign


def test_threshold_throws():
    alpha = 10.0
    x = torch.ones(10)

    with raises(ValueError):
        out = threshold(x, "noasd", alpha)


def test_threshold_backward():
    alpha = 10.0
    x = torch.ones(10)

    methods = [
        "super",
        "tanh",
        "tent",
        "circ",
    ]

    for method in methods:
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

    methods = [
        "super",
        "heaviside",
        "tanh",
        "tent",
        "circ",
    ]

    for method in methods:
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

    methods = [
        "super",
        "heaviside",
        "tanh",
        "tent",
        "circ",
    ]

    for method in methods:
        x = torch.ones(10)
        out = sign(x, method, alpha)
        assert torch.equal(out, torch.ones(10))

        x = torch.full((10,), 0.1)
        out = sign(x, method, alpha)
        assert torch.equal(out, torch.ones(10))

        x = torch.full((10,), -0.1)
        out = sign(x, method, alpha)
        assert torch.equal(out, -torch.ones(10))
