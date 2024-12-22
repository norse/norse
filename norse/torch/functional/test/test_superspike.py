import pytest
import platform

import torch
from norse.torch.functional.superspike import super_fn
from norse.torch.functional.heaviside import heaviside


def test_forward():
    assert torch.equal(super_fn(torch.ones(100), 100.0), heaviside(torch.ones(100)))
    assert torch.equal(super_fn(-1.0 * torch.ones(100), 100.0), torch.zeros(100))


def test_backward():
    x = torch.ones(10, requires_grad=True)
    out = super_fn(x, 100.0)
    out.backward(torch.ones(10))

    assert torch.sum(x.grad > 0) == 10

    x = torch.ones(10, requires_grad=True)
    out = super_fn(-0.001 * x, 100.0)
    out.backward(torch.ones(10))

    assert torch.sum(x.grad < 0) == 10


def test_backward_alpha():
    x = torch.ones(10, requires_grad=True)
    out = super_fn(x, 0.1)
    out.backward(torch.ones(10))
    expected = x / (0.1 * torch.abs(x) + 1.0).pow(2)
    assert torch.all(torch.eq(x.grad, expected))

    x.grad = None

    out = super_fn(x, 1000)
    out.backward(torch.ones(10))
    expected = x / (1000 * torch.abs(x) + 1.0).pow(2)
    assert torch.all(torch.eq(x.grad, expected))


class SomeClass(torch.nn.Module):
    def forward(self, x):
        return super_fn(x)


@pytest.mark.skipif(
    platform.system() == "Windows", reason="torch.compile not supported on Windows"
)
def test_compile():
    c = SomeClass()
    c = torch.compile(c)
    out = c(torch.ones(1, requires_grad=True))
    out.backward()
    assert out.sum() == 1
