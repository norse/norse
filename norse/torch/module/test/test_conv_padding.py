import pytest
import torch

from norse.torch.module.conv import LConv2d


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "padding, expected",
    [
        ("same", [[12.0, 21.0, 16.0], [27.0, 45.0, 33.0], [24.0, 39.0, 28.0]]),
        ("valid", [[45.0]]),
    ],
)
def test_lconv2d_string_padding(device, padding, expected):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    layer = LConv2d(1, 1, 3, padding=padding, device=device)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    values = torch.arange(1.0, 10.0, device=device).reshape(1, 1, 1, 3, 3)

    actual = layer(values)

    torch.testing.assert_close(
        actual, torch.tensor(expected, device=device)[None, None, None]
    )
