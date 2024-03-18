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


class SomeClass(torch.nn.Module):
    def forward(self, x):
        return super_fn(x)


@pytest.mark.skipif(
    not platform.system() == "Linux", reason="Only Linux supports torch.compile"
)
def test_compile():
    c = SomeClass()
    c = torch.compile(c)
    out = c(torch.ones(1, requires_grad=True))
    out.backward()
    assert out.sum() == 1
