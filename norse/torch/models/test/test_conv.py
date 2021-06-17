import torch
from norse.torch.models import conv


def test_convnet4():
    seq_length = 4
    batch_size = 2
    features = 1, 28, 28

    model = conv.ConvNet4()
    x = torch.randn(seq_length, batch_size, *features)

    out = model(x)
    assert out.shape == torch.Size([seq_length, batch_size, 10])


def test_convnet():
    seq_length = 4
    batch_size = 2
    features = 1, 28, 28

    model = conv.ConvNet()

    x = torch.randn(seq_length, batch_size, *features)
    out = model(x)
    assert out.shape == torch.Size([seq_length, batch_size, 10])
