import torch

from norse.torch.functional.lif import (
    LIFState,
    LIFFeedForwardState,
    lif_step,
    lif_feed_forward_step,
    lif_current_encoder,
)


def test_lif_step():
    x = torch.ones(20)
    s = LIFState(z=torch.zeros(10), v=torch.zeros(10), i=torch.zeros(10))
    input_weights = torch.linspace(0, 0.5, 200).view(10, 20)
    recurrent_weights = torch.linspace(0, -2, 100).view(10, 10)

    results = [
        torch.as_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        torch.as_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        torch.as_tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
        torch.as_tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
        torch.as_tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]),
        torch.as_tensor([0, 0, 0, 0, 1, 1, 0, 0, 0, 0]),
        torch.as_tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1]),
        torch.as_tensor([0, 0, 0, 1, 0, 1, 1, 0, 1, 1]),
        torch.as_tensor([0, 1, 1, 0, 1, 0, 0, 1, 1, 1]),
        torch.as_tensor([0, 0, 0, 0, 0, 1, 1, 0, 0, 1]),
    ]

    for result in results:
        z, s = lif_step(x, s, input_weights, recurrent_weights)
        assert torch.equal(z, result.float())


def test_lif_feed_forward_step():
    x = torch.ones(10)
    s = LIFFeedForwardState(v=torch.zeros(10), i=torch.zeros(10))

    results = [0.0, 0.1, 0.27, 0.487, 0.7335, 0.9963, 0.0, 0.3951, 0.7717, 0.0]

    for result in results:
        _, s = lif_feed_forward_step(x, s)
        assert torch.allclose(torch.as_tensor(result), s.v, atol=1e-4)


def test_lif_current_encoder():
    x = torch.ones(10)
    v = torch.zeros(10)

    results = [
        0.1,
        0.19,
        0.2710,
        0.3439,
        0.4095,
        0.4686,
        0.5217,
        0.5695,
        0.6126,
        0.6513,
    ]

    for result in results:
        _, v = lif_current_encoder(x, v)
        assert torch.allclose(torch.as_tensor(result), v, atol=1e-4)
