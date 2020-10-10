import torch
import norse.torch.models.mobilenet as mobilenet


def test_mobilenet_forward():
    model = mobilenet.mobilenet_v2()
    seq_length = 4
    batch_size = 2
    features = 3, 256, 256
    x = torch.randn(seq_length, batch_size, *features)
    out = model(x)
    assert out.shape == torch.Size([seq_length, batch_size, 1000])


def test_mobilenet_forward_pretrained():
    model = mobilenet.mobilenet_v2(pretrained=True)
    seq_length = 4
    batch_size = 2
    features = 3, 256, 256
    x = torch.randn(seq_length, batch_size, *features)
    out = model(x)
    assert out.shape == torch.Size([seq_length, batch_size, 1000])
