from typing import NamedTuple
import torch
from norse.torch.module import lif, snn


class MockParams(NamedTuple):
    my_param: int = -15


def test_snn_recurrent_cell_weights():
    in_w = torch.randn(3, 2)
    re_w = torch.randn(3, 3)
    n = snn.SNNRecurrentCell(
        None, None, 2, 3, p=MockParams(), input_weights=in_w, recurrent_weights=re_w
    )
    assert torch.all(torch.eq(n.input_weights, in_w))
    assert torch.all(torch.eq(n.recurrent_weights, re_w))


def test_snn_recurrent_weights():
    in_w = torch.randn(3, 2)
    re_w = torch.randn(3, 3)
    n = snn.SNNRecurrent(
        None, None, 2, 3, p=MockParams(), input_weights=in_w, recurrent_weights=re_w
    )
    assert torch.all(torch.eq(n.input_weights, in_w))
    assert torch.all(torch.eq(n.recurrent_weights, re_w))


def test_snn_cell_repr():
    n = snn.SNNCell(None, None, p=MockParams())
    assert str(n) == f"SNNCell(p={MockParams()}, dt=0.001)"
    n = lif.LIFCell(p=MockParams())
    assert str(n) == f"LIFCell(p={MockParams()}, dt=0.001)"


def test_snn_recurrent_cell_repr():
    n = snn.SNNRecurrentCell(None, None, 1, 2, MockParams())
    assert (
        str(n)
        == f"SNNRecurrentCell(input_size=1, hidden_size=2, p={MockParams()}, autapses=False, dt=0.001)"
    )
    n = lif.LIFRecurrentCell(1, 2, p=MockParams())
    assert (
        str(n)
        == f"LIFRecurrentCell(input_size=1, hidden_size=2, p={MockParams()}, autapses=False, dt=0.001)"
    )


def test_snn_repr():
    n = snn.SNN(None, None, p=MockParams())
    assert str(n) == f"SNN(p={MockParams()}, dt=0.001)"
    n = lif.LIF(p=MockParams())
    assert str(n) == f"LIF(p={MockParams()}, dt=0.001)"


def test_snn_recurrent_repr():
    n = snn.SNNRecurrent(None, None, 1, 2, MockParams())
    assert (
        str(n)
        == f"SNNRecurrent(input_size=1, hidden_size=2, p={MockParams()}, autapses=False, dt=0.001)"
    )
    n = lif.LIFRecurrent(1, 2, p=MockParams())
    assert (
        str(n)
        == f"LIFRecurrent(input_size=1, hidden_size=2, p={MockParams()}, autapses=False, dt=0.001)"
    )
