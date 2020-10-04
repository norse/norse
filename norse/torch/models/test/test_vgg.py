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


def forward(model):
    seq_length = 2
    batch_size = 1
    features = 3, 256, 256
    x = torch.randn(seq_length, batch_size, *features)
    out = model(x)
    assert out.shape == torch.Size([seq_length, batch_size, 1000])


def test_vgg11_bn():
    model = vgg.vgg11_bn()
    forward(model)


def test_vgg16():
    model = vgg.vgg16()
    forward(model)


def test_vgg16_bn():
    model = vgg.vgg16_bn()
    forward(model)


def test_vgg19():
    model = vgg.vgg19()
    forward(model)


def test_vgg19_bn():
    model = vgg.vgg19_bn()
    forward(model)


if __name__ == "__main__":
    test_vgg11_forward()
    test_vgg11_forward_pretrained()
