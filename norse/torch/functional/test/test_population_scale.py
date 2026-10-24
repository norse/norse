import pytest
import torch

from norse.torch.functional.encode import population_encode
from norse.torch.module.encode import PopulationEncoder


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("use_module", [False, True])
@pytest.mark.parametrize(
    "values, scale, expected",
    [
        (
            [0.0, 1.0],
            [1.0, 2.0],
            [[1.0, 0.8824969, 0.6065307], [0.3678794, 1.0, 0.3678794]],
        ),
        (
            [[0.0, 1.0], [0.0, 1.0]],
            [[1.0], [2.0]],
            [
                [[1.0, 0.8824969, 0.6065307], [0.6065307, 0.8824969, 1.0]],
                [[1.0, 0.3678794, 0.0183156], [0.3678794, 1.0, 0.3678794]],
            ],
        ),
    ],
    ids=["per-value", "per-batch-sample"],
)
def test_population_encode_tensor_scale(device, use_module, values, scale, expected):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    values = torch.tensor(values, device=device)
    scale = torch.tensor(scale, device=device)

    if use_module:
        actual = PopulationEncoder(3, scale=scale)(values)
    else:
        actual = population_encode(values, 3, scale=scale)

    torch.testing.assert_close(actual, torch.tensor(expected, device=device))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("scale", [0, 0.0, torch.tensor(0.0)])
def test_population_encode_zero_scale(device, scale):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    values = torch.tensor([0.0, 1.0], device=device)

    actual = population_encode(values, 3, scale=scale)

    torch.testing.assert_close(actual, torch.ones(2, 3, device=device))
