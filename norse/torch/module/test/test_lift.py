import pytest
import platform

import torch

from norse.torch.module.lif import LIF, LIFCell, LIFFeedForwardState
from norse.torch.module.lift import Lift
from norse.torch.module.sequential import SequentialState


def test_lift_conv():
    batch_size = 16
    seq_length = 20
    in_channels = 64
    out_channels = 32
    conv2d = Lift(torch.nn.Conv2d(in_channels, out_channels, 5, 1))
    data = torch.randn(seq_length, batch_size, in_channels, 20, 30)
    output = conv2d(data)

    assert output.shape == torch.Size([seq_length, batch_size, out_channels, 16, 26])


def test_lift_sequential():
    batch_size = 16
    seq_length = 20
    in_channels = 64
    out_channels = 32

    data = torch.randn(seq_length, batch_size, in_channels, 20, 30)
    module = torch.nn.Sequential(
        Lift(torch.nn.Conv2d(in_channels, out_channels, 5, 1)),
        LIF(),
    )
    output, _ = module(data)

    assert output.shape == torch.Size([seq_length, batch_size, out_channels, 16, 26])


def test_lift_stateful():
    c = Lift(LIFCell())
    data = torch.randn(5, 2)
    out = c(data)
    assert type(out) == tuple
    assert out[0].shape == (5, 2)
    assert type(out[1]) == LIFFeedForwardState


def test_lift_sequential_stateful():
    c = Lift(SequentialState(LIFCell()))
    data = torch.randn(5, 2)
    out = c(data)
    assert type(out) == tuple
    assert out[0].shape == (5, 2)
    assert type(out[1]) == list
    assert type(out[1][0]) == LIFFeedForwardState


@pytest.mark.skipif(
    not platform.system() == "Linux", reason="Only Linux supports torch.compile"
)
def test_compile_lift():
    c = Lift(LIFCell())
    c = torch.compile(c, mode="reduce-overhead")
    data = torch.randn(5, 2)
    out = c(data)
    assert type(out) == tuple


@pytest.mark.skipif(
    not torch.cuda.is_available() or not platform.system() == "Linux",
    reason="no cuda device or not on linux",
)
def test_compile_lift():
    c = Lift(LIFCell()).cuda()
    c = torch.compile(c, mode="reduce-overhead")
    data = torch.randn(5, 2).cuda()
    out = c(data)
    assert type(out) == tuple
    assert out[0].device.type == "cuda"


if __name__ == "__main__":
    test_lift_conv()
    test_lift_sequential()
