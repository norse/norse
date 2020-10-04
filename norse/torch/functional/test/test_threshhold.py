import torch

from norse.torch.functional.threshold import threshold, sign


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
