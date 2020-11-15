import torch
import pytest

import norse.torch.models.transformer as transformer


def test_encode_layer_forward():
    model = transformer.SpikingTransformerEncoderLayer(10, 10)
    seq_length = 4
    batch_size = 2
    x = torch.randn(seq_length, batch_size, 10)
    out = model(x)
    assert out.shape == torch.Size([seq_length, batch_size, 10])
