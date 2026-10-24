import pytest
import torch

from norse.torch.functional.encode import (
    constant_current_lif_encode,
    poisson_encode_step,
    signed_poisson_encode_step,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("current, expected", [(0.0, [0.0, 0.0]), (16.0, [1.0, 1.0])])
def test_constant_current_encoder_accepts_scalar(device, current, expected):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    actual = constant_current_lif_encode(torch.tensor(current, device=device), 2)

    torch.testing.assert_close(actual, torch.tensor(expected, device=device))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("value", [0.0, 1.0])
def test_poisson_step_accepts_scalar(device, value):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    actual = poisson_encode_step(torch.tensor(value, device=device))

    torch.testing.assert_close(actual, torch.tensor(value, device=device))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("value", [-1.0, 0.0, 1.0])
def test_signed_poisson_step_accepts_scalar(device, value):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    actual = signed_poisson_encode_step(torch.tensor(value, device=device))

    torch.testing.assert_close(actual, torch.tensor(value, device=device))
