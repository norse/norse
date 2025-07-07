from dataclasses import dataclass
from numbers import Number
from typing import Tuple
import platform
from weakref import ref

import torch
import onnx

import pytest

import norse.torch.utils.pytree as pytree

# pytype: disable=wrong-arg-count,wrong-keyword-args,unsupported-operands


class MockState(pytree.StateTuple, metaclass=pytree.MultipleInheritanceNamedTupleMeta):
    x: torch.Tensor
    y: torch.Tensor = torch.randn(10, 2)
    z: float = 1.28


class MockModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.s = MockState(torch.randn(10, 2))

    def forward(self, x, s: MockState):
        return x + self.s.x + self.s.y + s[0] + s[1]


def test_state_create():
    x = torch.randn(10, 2)
    s = MockState(x=x)
    assert torch.all(torch.eq(s.x, x))
    assert s.y.shape == (10, 2)
    assert s.z == 1.28


def test_state_is_tuple():
    s = MockState(torch.randn(1))
    assert isinstance(s, Tuple)


def test_state_clone():
    x = torch.tensor([1.0, 2.0])
    s1 = MockState(x=x)
    s2 = MockState(x=s1.x + 1.0)
    assert torch.allclose(s1.x, s2.x - 1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device available")
def test_to_device():
    try:
        r = torch.randn(2)
        s = MockState(r)
        sg = s.to("cuda")
        assert sg.x.device.type == "cuda"
        assert sg.y.device.type == "cuda"
    except (RuntimeError, AssertionError):
        pass  # Ignore non-cuda systems

    s = MockState(torch.randn(2))
    s.cpu()
    assert s.x.device.type == "cpu"
    assert s.y.device.type == "cpu"


def test_to_float():
    s = MockState(torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32))
    assert s.x.dtype == torch.int32
    assert s.y.dtype == torch.int32
    s2 = s.float()
    assert s2.x.dtype == torch.float32
    assert s2.y.dtype == torch.float32


def test_to_int():
    s = MockState(torch.randn(2))
    assert s.x.dtype == torch.float32
    assert s.y.dtype == torch.float32
    s2 = s.int()
    assert s2.x.dtype == torch.int32
    assert s2.y.dtype == torch.int32


def test_broadcast():
    s = MockState(torch.randn(2))
    assert s.x.shape == (2,)

    with pytest.raises(ValueError):
        s.broadcast(torch.zeros(10, 2))

    # Single-value tensor
    s = MockState(torch.randn(1), torch.randn(1))
    s2 = s.broadcast(torch.zeros(3, 2))
    assert s2.x.shape == (3, 2)
    assert s2.y.shape == (3, 2)

    # Scalar
    s = MockState(0, 1)
    assert s.x == 0
    assert s.y == 1
    s3 = s.broadcast(torch.zeros(2, 1))
    assert isinstance(s3.x, Number)
    assert isinstance(s3.y, Number)


def test_onnx():
    m = MockModule()
    s = MockState(torch.randn(10, 2))
    torch.onnx.export(m, (torch.randn(10, 2), s), "pytree.onnx")
    loaded = onnx.load("pytree.onnx")
    onnx.checker.check_model(loaded)


@pytest.mark.skipif(
    platform.system() == "Windows", reason="torch.compile not supported on Windows"
)
def test_compile():
    class MockModule(torch.nn.Module):
        def __init__(self, p: MockState):
            super().__init__()
            self.p = p

        def forward(
            self, x: torch.Tensor, s: MockState
        ) -> Tuple[torch.Tensor, MockState]:
            y = x + s.x * self.p.y.sum()
            return y, MockState(y)

    s = MockState(torch.ones(1), y=torch.ones(2, 2))
    m = MockModule(s)
    m = torch.compile(m)
    y1, s = m(torch.ones(1), s)
    y2, s = m(torch.ones(1), s)
    assert isinstance(s, MockState)
    assert torch.eq(y1, torch.tensor([5]))
    assert torch.eq(y2, torch.tensor([5 + 16]))
