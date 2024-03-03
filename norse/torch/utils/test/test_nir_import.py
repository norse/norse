import nir
import nirtorch
import numpy as np
import torch

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
    m = _convert_nodes(nir.Conv2d(None, w, 1, 0, 1, 1, b))
    assert isinstance(m.conv2d, torch.nn.Conv2d)
    x = torch.randn(1, 2, 3, 3)
    out = m(x)
    m(torch.randn(1, 2, 3, 3))  # Test application


def test_import_flatten():
    m = _convert_nodes(nir.Flatten((1, 2, 3, 3), 1, -1))
    assert isinstance(m.flatten, torch.nn.Flatten)
    assert m(torch.randn(1, 2, 3, 3))[0].shape == (1, 18)


def test_import_if():
    m = _convert_nodes(nir.IF(torch.randn(10), torch.randn(10)))
    assert isinstance(getattr(m, "if"), norse.IAFCell)
    m(torch.randn(1, 10))  # Test application


def test_import_lif():
    m = _convert_nodes(
        nir.LIF(torch.randn(10), torch.ones(10), torch.randn(10), torch.randn(10))
    )
    assert isinstance(m.lif, norse.LIFBoxCell)
    m(torch.randn(1, 10))  # Test application


def test_import_cubalif():
    orig = nir.CubaLIF(
        torch.randn(10),
        torch.randn(10),
        torch.ones(10),
        torch.randn(10),
        torch.randn(10),
    )
    m = _convert_nodes(orig)
    assert isinstance(m.cubalif, norse.utils.import_nir.CubaLIF)
    assert isinstance(m.cubalif.synapse, norse.LIBoxCell)
    assert isinstance(m.cubalif.lif, norse.LIFBoxCell)
    m(torch.randn(1, 10))  # Test application
    torch.allclose(orig.tau_mem, 1000 / m.cubalif.lif.p.tau_mem_inv)


def test_import_sumpool2d():
    m = _convert_nodes(
        nir.SumPool2d(np.array([3, 3]), np.array([1, 1]), np.array([0, 0]))
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
