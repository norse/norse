import pytest

import torch

import norse.torch.utils.clone as clone


def test_clone_number():
    x = 1.0
    y = clone.clone_tensor(x)
    assert isinstance(y, torch.Tensor)
    assert y.item() == x


def test_clone_tensor():
    x = torch.tensor(1.0)
    y = clone.clone_tensor(x)
    assert isinstance(y, torch.Tensor)
    assert y.item() == x.item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")
def test_clone_retain_device():
    x = torch.tensor(1.0, device="cuda")
    y = clone.clone_tensor(x)
    assert isinstance(y, torch.Tensor)
    assert y.item() == x.item()
    assert y.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")
def test_clone_tensor_device():
    x = torch.tensor(1.0)
    y = clone.clone_tensor(x, "cuda")
    assert isinstance(y, torch.Tensor)
    assert y.item() == x.item()
    assert y.device.type == "cuda"


def test_clone_jit():
    def f(x):
        return clone.clone_tensor(x) + 1

    m = torch.jit.script(f)
    y = m(torch.tensor(1.0))
    assert y.item() == 2.0
