import torch

from norse.torch.functional.cuba_lif import CubaLIFFeedForwardState
from norse.torch.module.cuba_lif import CubaLIF, CubaLIFCell


def test_cuba_lif_cell_feed_forward_step_batch():
    x = torch.ones(2, 1)
    s = CubaLIFFeedForwardState(v=torch.zeros(2, 1), i=torch.zeros(2, 1))

    z, s = CubaLIFCell()(x, s)
    assert z.shape == (2, 1)
    assert torch.allclose(s.i, torch.ones(2, 1))
    assert torch.allclose(s.v, torch.full((2, 1), 0.1))


def test_cuba_lif_cell_default_state():
    z, s = CubaLIFCell()(torch.ones(3, 4))
    assert z.shape == (3, 4)
    assert s.v.shape == (3, 4)
    assert s.i.shape == (3, 4)


def test_cuba_lif_sequence():
    x = torch.ones(5, 2, 4)
    cell = CubaLIFCell()
    s = None
    expected = []
    for xt in x:
        zt, s = cell(xt, s)
        expected.append(zt)

    z, s_seq = CubaLIF()(x)
    assert z.shape == (5, 2, 4)
    assert torch.allclose(z, torch.stack(expected))
    assert torch.allclose(s_seq.v, s.v)


def test_cuba_lif_backward():
    x = torch.ones(10, 2, 4, requires_grad=True)
    z, _ = CubaLIF()(x)
    z.sum().backward()
    assert x.grad is not None
