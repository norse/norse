import torch

from norse.torch.module.lif import LIFFeedForwardLayer
from norse.torch.module.lift import Lift


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
        LIFFeedForwardLayer(),
    )
    output, _ = module(data)

    assert output.shape == torch.Size([seq_length, batch_size, out_channels, 16, 26])


if __name__ == "__main__":
    test_lift_conv()
    test_lift_sequential()
