import torch

import norse
from norse.torch.functional.lif import (
    LIFState,
    LIFFeedForwardState,
    LIFParameters,
    LIFParametersJIT,
    _lif_step_jit,
    lif_step,
    lif_step_integral,
    lif_feed_forward_step,
    lif_feed_forward_integral,
    _lif_feed_forward_step_jit,
    _lif_feed_forward_integral_jit,
    lif_current_encoder,
)


def test_lif_cpp_back():
    x = torch.ones(2)
    s = LIFState(z=torch.zeros(1), v=torch.zeros(1), i=torch.zeros(1))
    s.v.requires_grad = True
    input_weights = torch.ones(2)
    recurrent_weights = torch.ones(1)
    _, s = lif_step(x, s, input_weights, recurrent_weights)
    z, s = lif_step(x, s, input_weights, recurrent_weights)
    z.sum().backward()


def test_lif_jit_back():
    x = torch.ones(2, requires_grad=True)
    s = LIFState(z=torch.zeros(1), v=torch.zeros(1), i=torch.zeros(1))
    s.v.requires_grad = True
    input_weights = torch.ones(2)
    recurrent_weights = torch.ones(1)
    p = LIFParameters()
    jit_params = LIFParametersJIT(
        tau_syn_inv=p.tau_syn_inv,
        tau_mem_inv=p.tau_mem_inv,
        v_leak=p.v_leak,
        v_th=p.v_th,
        v_reset=p.v_reset,
        method=p.method,
        alpha=torch.as_tensor(p.alpha),
    )
    _, s = _lif_step_jit(x, s, input_weights, recurrent_weights, p=jit_params)
    z, s = _lif_step_jit(x, s, input_weights, recurrent_weights, p=jit_params)
    print(x.shape, z.shape, input_weights.shape, recurrent_weights.shape)
    z.sum().backward()


def test_lif_heavi():
    x = torch.ones(2, 1)
    s = LIFState(z=torch.ones(2, 1), v=torch.zeros(2, 1), i=torch.zeros(2, 1))
    input_weights = torch.ones(1, 1) * 10
    recurrent_weights = torch.ones(1, 1)
    p = LIFParameters(method="heaviside")
    _, s = lif_step(x, s, input_weights, recurrent_weights, p)
    z, s = lif_step(x, s, input_weights, recurrent_weights, p)
    assert z.max() > 0
    assert z.shape == (2, 1)


def test_lif_feed_forward_step():
    x = torch.ones(10)
    s = LIFFeedForwardState(v=torch.zeros(10), i=torch.zeros(10))

    results = [0.0, 0.1, 0.27, 0.487, 0.7335, 0.9963, 0.0, 0.3951, 0.7717, 0.0]

    for result in results:
        _, s = lif_feed_forward_step(x, s)
        assert torch.allclose(torch.as_tensor(result), s.v, atol=1e-4)


def test_lif_feed_forward_step_batch():
    x = torch.ones(2, 1)
    s = LIFFeedForwardState(v=torch.zeros(2, 1), i=torch.zeros(2, 1))

    z, s = lif_feed_forward_step(x, s)
    assert z.shape == (2, 1)


def test_lif_feed_forward_step_jit():
    x = torch.ones(10)
    s = LIFFeedForwardState(v=torch.zeros(10), i=torch.zeros(10))

    p = LIFParametersJIT(
        torch.as_tensor(1.0 / 5e-3),
        torch.as_tensor(1.0 / 1e-2),
        torch.as_tensor(0.0),
        torch.as_tensor(1.0),
        torch.as_tensor(0.0),
        "super",
        torch.as_tensor(0.0),
    )

    results = [0.0, 0.1, 0.27, 0.487, 0.7335, 0.9963, 0.0, 0.3951, 0.7717, 0.0]

    for result in results:
        _, s = _lif_feed_forward_step_jit(x, s, p)
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
