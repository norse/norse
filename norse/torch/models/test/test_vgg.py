import pytest
import torch

from norse.torch.models import vgg
from norse.torch.models.test.conftest import local_state_dict


def test_vgg11_forward():
    model = vgg.vgg11()
    print(model)
    seq_length = 1
    batch_size = 2
    features = 3, 256, 256
    x = torch.randn(seq_length, batch_size, *features)
    out = model(x)
    assert out.shape == torch.Size([seq_length, batch_size, 1000])


def test_vgg11_forward_pretrained(monkeypatch):
    monkeypatch.setattr(
        vgg, "load_state_dict_from_url", lambda *a, **kw: local_state_dict("vgg11")
    )
    model = vgg.vgg11(pretrained=True)
    seq_length = 1
    batch_size = 2
    features = 3, 256, 256
    x = torch.randn(seq_length, batch_size, *features)
    out = model(x)
    assert out.shape == torch.Size([seq_length, batch_size, 1000])


def forward(model):
    seq_length = 1
    batch_size = 1
    features = 3, 256, 256
    x = torch.randn(seq_length, batch_size, *features)
    out = model(x)
    assert out.shape == torch.Size([seq_length, batch_size, 1000])


def test_vgg11_bn():
    model = vgg.vgg11_bn()
    assert isinstance(model, vgg.VGG)


def test_vgg13():
    model = vgg.vgg13()
    assert isinstance(model, vgg.VGG)


def test_vgg13_bn():
    model = vgg.vgg13_bn()
    assert isinstance(model, vgg.VGG)


def test_vgg16():
    model = vgg.vgg16()
    assert isinstance(model, vgg.VGG)


def test_vgg16_bn():
    model = vgg.vgg16_bn()
    assert isinstance(model, vgg.VGG)


def test_vgg19():
    model = vgg.vgg19()
    assert isinstance(model, vgg.VGG)


def test_vgg19_bn():
    model = vgg.vgg19_bn()
    assert isinstance(model, vgg.VGG)


@pytest.mark.download
def test_vgg11_forward_pretrained_download():
    """The same path as test_vgg11_forward_pretrained, against the real weights."""
    model = vgg.vgg11(pretrained=True)
    x = torch.randn(1, 2, 3, 256, 256)
    assert model(x).shape == torch.Size([1, 2, 1000])
