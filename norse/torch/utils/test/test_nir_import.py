import nir
import nirtorch
import numpy as np
import torch

from norse.torch.functional.scale_space import exponential_tau_mem_inv
from norse.torch.utils.import_nir import _to_tensor
import norse.torch as norse


def _convert_nodes(*args):
    return norse.from_nir(nir.NIRGraph.from_list(*args))


def test_convert_to_tensor():
    a = np.random.randn(2, 3).astype(np.float32)
    b = torch.tensor(a)
    assert torch.allclose(_to_tensor(a), b)

    a = torch.tensor([0.3, -12])
    assert torch.allclose(_to_tensor(a), a)


def test_import_linear():
    w = np.random.randn(2, 3)
    b = np.random.randn(2)
    n = nir.NIRGraph.from_list(
        nir.Input({"input": torch.tensor([3])}),
        nir.Affine(w, b),
        nir.Output({"output": torch.tensor([2])}),
    )
    m = norse.from_nir(n)
    assert isinstance(m.affine, torch.nn.Linear)
    x = torch.randn(1, 3)
    out, _ = m(x)
    actual = x @ _to_tensor(w).T + _to_tensor(b)
    assert torch.allclose(out, actual)


def test_import_conv2d():
    w = np.random.randn(1, 2, 3, 3).astype(np.float32)
    b = np.random.randn(1).astype(np.float32)
    conv = torch.nn.Conv2d(2, 1, 3)
    conv.weight.data = _to_tensor(w)
    conv.bias.data = _to_tensor(b)
    m = _convert_nodes(nir.Conv2d((5, 5), w, 1, 0, 1, 1, b))
    assert isinstance(m.conv2d, torch.nn.Conv2d)
    x = torch.randn(1, 2, 3, 3)
    out = m(x)
    m(torch.randn(1, 2, 3, 3))  # Test application


def test_import_flatten():
    m = _convert_nodes(nir.Flatten((1, 2, 3, 3), 1, -1))
    assert isinstance(m.flatten, torch.nn.Flatten)
    assert m(torch.randn(1, 2, 3, 3))[0].shape == (1, 18)


def test_import_if():
    m = _convert_nodes(nir.IF(torch.randn(10), torch.randn(10), torch.randn(10)))
    assert isinstance(getattr(m, "nir_node_if"), norse.IAFCell)
    m(torch.randn(1, 10))  # Test application


def test_import_lif():
    m = _convert_nodes(
        nir.LIF(
            torch.rand(10) + 0.01,  # tau must be a positive time constant
            torch.ones(10),
            torch.randn(10),
            torch.randn(10),
            torch.randn(10),
        )
    )
    assert isinstance(m.lif, norse.LIFBoxCell)
    m(torch.randn(1, 10))  # Test application


def test_import_li():
    tau = np.ones(10) * 0.01
    r = np.ones(10)
    v_leak = np.random.randn(10).astype(np.float32)

    m = _convert_nodes(nir.LI(tau, r, v_leak))
    assert isinstance(m.li, norse.LIBoxCell)

    x = torch.randn(1, 10)
    output, _ = m(x)
    assert output.shape == (1, 10)

    # NIR's tau is a continuous time constant, so the imported integrator is
    # configured to decay by exactly exp(-dt/tau) per step.
    dt = 0.001
    actual_tau_mem_inv = m.li.p.tau_mem_inv.detach().numpy()
    assert np.allclose(
        exponential_tau_mem_inv(torch.as_tensor(tau), dt).numpy(),
        actual_tau_mem_inv,
        rtol=1e-6,
    )
    assert np.allclose(1 - dt * actual_tau_mem_inv, np.exp(-dt / tau), atol=1e-6)

    actual_v_leak = m.li.p.v_leak.detach().numpy()
    assert np.allclose(v_leak, actual_v_leak)


def test_import_cubalif():
    dt = 0.001
    orig = nir.CubaLIF(
        torch.rand(10) + 0.01,  # tau_syn, a positive time constant
        torch.rand(10) + 0.01,  # tau_mem, a positive time constant
        torch.ones(10),
        torch.randn(10),
        torch.randn(10),
        torch.randn(10),
    )
    m = _convert_nodes(orig)
    assert isinstance(m.cubalif, norse.utils.import_nir.CubaLIF)
    assert isinstance(m.cubalif.synapse, norse.LIBoxCell)
    assert isinstance(m.cubalif.lif, norse.LIFBoxCell)
    m(torch.randn(1, 10))  # Test application
    assert torch.allclose(
        exponential_tau_mem_inv(orig.tau_mem, dt),
        m.cubalif.lif.p.tau_mem_inv,
        rtol=1e-6,
    )
    assert torch.allclose(
        exponential_tau_mem_inv(orig.tau_syn, dt),
        m.cubalif.synapse.p.tau_mem_inv,
        rtol=1e-6,
    )


def test_import_sumpool2d():
    m = _convert_nodes(
        nir.Input({"input": torch.tensor([5, 5])}),
        nir.SumPool2d(np.array([3, 3]), np.array([1, 1]), np.array([0, 0])),
    )
    assert isinstance(m.sumpool2d, torch.nn.LPPool2d)
    assert m(torch.randn(1, 2, 3, 3))[0].shape == (1, 2, 1, 1)


def test_import_recurrent():
    m = norse.from_nir("norse/torch/utils/test/braille.nir")
    data = torch.ones(8, 12)
    assert m(data)[0].shape == (
        8,
        7,
    )
