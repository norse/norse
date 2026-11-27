import pytest
import torch

from norse.torch.models import mobilenet
from norse.torch.models.test.conftest import local_state_dict


def test_mobilenet_forward():
    model = mobilenet.mobilenet_v2()
    seq_length = 4
    batch_size = 2
    features = 3, 256, 256
    x = torch.randn(seq_length, batch_size, *features)
    out = model(x)
    assert out.shape == torch.Size([seq_length, batch_size, 1000])


def test_mobilenet_forward_pretrained(monkeypatch):
    monkeypatch.setattr(
        mobilenet,
        "load_state_dict_from_url",
        lambda *a, **kw: local_state_dict("mobilenet_v2"),
    )
    model = mobilenet.mobilenet_v2(pretrained=True)
    seq_length = 4
    batch_size = 2
    features = 3, 256, 256
    x = torch.randn(seq_length, batch_size, *features)
    out = model(x)
    assert out.shape == torch.Size([seq_length, batch_size, 1000])


def test_mobilenet_raises():
    with pytest.raises(ValueError):
        _ = mobilenet.MobileNetV2(inverted_residual_setting=[])


def test_mobilenet_no_defaults():
    model = mobilenet.MobileNetV2(
        width_mult=0.1,
        round_nearest=4,
        block=mobilenet.InvertedResidual,
        norm_layer=torch.nn.BatchNorm2d,
    )
    assert model.last_channel % 4 == 0


@pytest.mark.download
def test_mobilenet_forward_pretrained_download():
    """The same path as test_mobilenet_forward_pretrained, against the real weights."""
    model = mobilenet.mobilenet_v2(pretrained=True)
    x = torch.randn(4, 2, 3, 256, 256)
    assert model(x).shape == torch.Size([4, 2, 1000])
