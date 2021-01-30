import torch
import pytest
from norse.torch.models import mobilenet


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
