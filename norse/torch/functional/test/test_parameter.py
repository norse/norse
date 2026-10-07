import pytest
import torch

from norse.torch.functional.parameter import _float_parameter


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_integer_parameter_matches_scalar_promotion(dtype):
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        value = torch.tensor([42, 200])
        torch.testing.assert_close(0.001 * _float_parameter(value), 0.001 * value)
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_float_parameter_preserves_gradient(dtype):
    value = torch.tensor(200.0, dtype=dtype, requires_grad=True)
    converted = _float_parameter(value)
    assert converted is value
    (converted * 0.001).backward()
    torch.testing.assert_close(value.grad, torch.tensor(0.001, dtype=dtype))


def test_float_parameter_preserves_python_scalar():
    assert _float_parameter(250.0) == 250.0
