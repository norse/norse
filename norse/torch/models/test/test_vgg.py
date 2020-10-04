import torch
import norse.torch.models.vgg as vgg


def test_vgg11_forward():
    model = vgg.vgg11()
    print(model)
    seq_length = 5
    batch_size = 2
    features = 3, 256, 256
    x = torch.randn(seq_length, batch_size, *features)
    out = model(x)
    assert out.shape == torch.Size([seq_length, batch_size, 1000])


def test_vgg11_forward_pretrained():
    model = vgg.vgg11(pretrained=True)
    print(model)
    seq_length = 5
    batch_size = 2
    features = 3, 256, 256
    x = torch.randn(seq_length, batch_size, *features)
    out = model(x)
    assert out.shape == torch.Size([seq_length, batch_size, 1000])


if __name__ == "__main__":
    test_vgg11_forward()
    test_vgg11_forward_pretrained()
