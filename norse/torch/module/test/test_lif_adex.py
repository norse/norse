import torch

from norse.torch.module.lif_adex import (
    LIFAdExCell,
    LIFAdExLayer,
    LIFAdExFeedForwardCell,
    LIFAdExFeedForwardState,
)


def test_lif_adex_cell():
    cell = LIFAdExCell(2, 4)
    data = torch.randn(5, 2)
    out, _ = cell(data)

    assert out.shape == (5, 4)


def test_lif_adex_repr():
    cell = LIFAdExCell(2, 4)
    assert (
        str(cell)
        == "LIFAdExCell(2, 4, p=LIFAdExParameters(adaptation_current=tensor(4), adaptation_spike=tensor(0.0200), delta_T=tensor(0.5000), tau_ada_inv=tensor(2.), tau_syn_inv=tensor(200.), tau_mem_inv=tensor(100.), v_leak=tensor(0.), v_th=tensor(1.), v_reset=tensor(0.), method='super', alpha=0.0), dt=0.001)"
    )


def test_lif_adex_cell_backward():
    cell = LIFAdExCell(2, 4)
    data = torch.randn(5, 2)
    out, _ = cell(data)
    out.sum().backward()


def test_lif_adex_layer():
    layer = LIFAdExLayer(2, 4)
    data = torch.randn(10, 5, 2)
    out, _ = layer(data)

    assert out.shape == (10, 5, 4)


def test_lif_adex_feedforward_cell():
    layer = LIFAdExFeedForwardCell()
    data = torch.randn(5, 2, 4)
    out, _ = layer(data)

    assert out.shape == (5, 2, 4)


def test_lif_adex_feedforward_cell_state():
    cell = LIFAdExFeedForwardCell()
    input_tensor = torch.randn(5, 2, 4)

    state = LIFAdExFeedForwardState(
        v=cell.p.v_leak,
        i=torch.zeros(
            input_tensor.shape,
        ),
        a=torch.zeros(
            input_tensor.shape,
        ),
    )

    out, _ = cell(input_tensor, state)

    assert out.shape == (5, 2, 4)


def test_lif_adex_feedforward_repr():
    cell = LIFAdExFeedForwardCell()
    assert (
        str(cell)
        == "LIFAdExFeedForwardCell(p=LIFAdExParameters(adaptation_current=tensor(4), adaptation_spike=tensor(0.0200), delta_T=tensor(0.5000), tau_ada_inv=tensor(2.), tau_syn_inv=tensor(200.), tau_mem_inv=tensor(100.), v_leak=tensor(0.), v_th=tensor(1.), v_reset=tensor(0.), method='super', alpha=0.0), dt=0.001)"
    )


def test_lif_adex_feedforward_cell_backward():
    cell = LIFAdExFeedForwardCell()
    data = torch.randn(5, 2, 4)
    out, _ = cell(data)
    out.sum().backward()
