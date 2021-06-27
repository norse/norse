import sys

import torch
import pytest

import norse
from norse.torch.functional.lif import (
    LIFState,
    LIFFeedForwardState,
    LIFParameters,
    LIFParametersJIT,
    lif_step,
    lif_step_integral,
    lif_feed_forward_step,
    lif_feed_forward_integral,
    _lif_feed_forward_step_jit,
    lif_current_encoder,
)


@pytest.fixture(autouse=True)
def cpp_fixture():
    norse.utils.IS_OPS_LOADED = True  # Enable cpp


@pytest.fixture()
def jit_fixture():
    norse.utils.IS_OPS_LOADED = False  # Disable cpp


def test_lif_cpp_and_jit_step():
    assert norse.utils.IS_OPS_LOADED
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

    cpp_results = []
    cpp_states = []
    for result in results:
        z, s = lif_step(x, s, input_weights, recurrent_weights)
        cpp_results.append(z)
        cpp_states.append(s)

    s = LIFState(z=torch.zeros(10), v=torch.zeros(10), i=torch.zeros(10))
    norse.utils.IS_OPS_LOADED = False  # Disable cpp

    for i, result in enumerate(results):
        z, s = lif_step(x, s, input_weights, recurrent_weights)
        assert torch.equal(z, result.float())
        assert torch.equal(z, cpp_results[i])
        assert torch.equal(s.v, cpp_states[i].v)
        assert torch.equal(s.z, cpp_states[i].z)
        assert torch.equal(s.i, cpp_states[i].i)


def test_lif_cpp_back(cpp_fixture):
    x = torch.ones(2)
    s = LIFState(z=torch.zeros(1), v=torch.zeros(1), i=torch.zeros(1))
    s.v.requires_grad = True
    input_weights = torch.ones(2)
    recurrent_weights = torch.ones(1)
    _, s = lif_step(x, s, input_weights, recurrent_weights)
    z, s = lif_step(x, s, input_weights, recurrent_weights)
    z.sum().backward()


def test_lif_jit_back(jit_fixture):
    x = torch.ones(2)
    s = LIFState(z=torch.zeros(1), v=torch.zeros(1), i=torch.zeros(1))
    s.v.requires_grad = True
    input_weights = torch.ones(2)
    recurrent_weights = torch.ones(1)
    _, s = lif_step(x, s, input_weights, recurrent_weights)
    z, s = lif_step(x, s, input_weights, recurrent_weights)
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


def test_lif_integral_jit(jit_fixture):
    x = torch.ones(10, 20)
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

    s = LIFState(z=torch.zeros(10), v=torch.zeros(10), i=torch.zeros(10))
    norse.utils.IS_OPS_LOADED = False  # Disable cpp

    z, s = lif_step_integral(x, s, input_weights, recurrent_weights)
    assert torch.all(torch.equal(z, results))


def test_lif_integral_cpp(cpp_fixture):
    x = torch.ones(10, 20)
    s = LIFState(z=torch.zeros(20), v=torch.zeros(20), i=torch.zeros(20))
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

    z, s = lif_step_integral(x, s, input_weights, recurrent_weights)
    assert torch.all(torch.equal(z.size(), torch.tensor([10, 20])))
    assert torch.all(torch.equal(s.z.size(), torch.tensor([20])))
    assert torch.all(torch.equal(s.v.size(), torch.tensor([20])))
    assert torch.all(torch.equal(s.i.size(), torch.tensor([20])))
    assert torch.all(torch.equal(z, results))


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


def test_lif_feed_forward_step_cpp(cpp_fixture):
    assert norse.utils.IS_OPS_LOADED == True
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


def test_lif_feed_forward_step_jit(jit_fixture):
    assert norse.utils.IS_OPS_LOADED == False
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


def test_lif_feed_forward_integrate_cpp(cpp_fixture):
    assert norse.utils.IS_OPS_LOADED == True
    x = torch.ones(9, 2)
    s = LIFFeedForwardState(v=torch.zeros(2), i=torch.zeros(2))

    expected_v = torch.tensor(0.7717)

    _, s = lif_feed_forward_integral(x, s)
    assert torch.allclose(expected_v, s.v[0], atol=1e-4)


def test_lif_feed_forward_integrate_jit(jit_fixture):
    x = torch.ones(9, 2)
    s = LIFFeedForwardState(v=torch.zeros(2), i=torch.zeros(2))

    expected_v = torch.tensor(0.7717)

    _, s = lif_feed_forward_integral(x, s)
    assert torch.allclose(expected_v, s.v[0], atol=1e-4)


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
