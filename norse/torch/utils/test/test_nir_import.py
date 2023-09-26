import nir
import nirtorch
import torch

import norse.torch as norse


def _convert_nodes(*args):
    return norse.from_nir(nir.NIRGraph.from_list(*args))


def test_import_linear():
    n = nir.NIRGraph.from_list(
        nir.Input({"input": torch.tensor([2])}),
        nir.Affine(torch.randn(2, 3), torch.randn(2)),
        nir.Output({"output": torch.tensor([3])}),
    )
    m = norse.from_nir(n)
    assert isinstance(m.affine, torch.nn.Linear)
    m(torch.randn(1, 3))  # Test application


def test_import_conv2d():
    m = _convert_nodes(nir.Conv2d(torch.randn(1, 2, 3, 3), 1, 0, 1, 1, torch.randn(2)))
    assert isinstance(m.conv2d, torch.nn.Conv2d)
    m(torch.randn(1, 2, 3, 3))  # Test application


def test_import_if():
    m = _convert_nodes(nir.IF(torch.randn(10), torch.randn(10)))
    assert isinstance(getattr(m, "if"), norse.IAFCell)
    m(torch.randn(1, 10))  # Test application

def test_import_lif():
    m = _convert_nodes(nir.LIF(torch.randn(10), torch.ones(10), torch.randn(10), torch.randn(10)))
    assert isinstance(m.lif, norse.LIFBoxCell)
    m(torch.randn(1, 10))  # Test application

def test_import_cubalif():
    m = _convert_nodes(nir.CubaLIF(torch.randn(10), torch.randn(10), torch.ones(10), torch.randn(10), torch.randn(10)))
    assert isinstance(m.cubalif, norse.LIFCell)
    m(torch.randn(1, 10))  # Test application