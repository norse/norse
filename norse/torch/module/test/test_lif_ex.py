import torch

from norse.torch.module.lif_ex import (
    LIFExCell,
    LIFExLayer,
    LIFExState,
    LIFExFeedForwardCell,
    LIFExFeedForwardState,
)


def test_lif_ex_cell():
    cell = LIFExCell(2, 4)
    data = torch.randn(5, 2)
    out, _ = cell(data)

    assert out.shape == (5, 4)


def test_lif_ex_cell_backward():
    cell = LIFExCell(2, 4)
    data = torch.randn(5, 2)
    out, _ = cell(data)
    out.sum().backward()


def test_lif_ex_layer():
    layer = LIFExLayer(2, 4)
    data = torch.randn(10, 5, 2)
    out, _ = layer(data)

    assert out.shape == (10, 5, 4)


def test_lif_ex_layer_state():
    layer = LIFExLayer(2, 4)
    input_tensor = torch.randn(10, 5, 2)

    state = LIFExState(
        z=torch.zeros(
            (input_tensor.shape[1], layer.cell.hidden_size),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        ),
        v=layer.cell.p.v_leak,
        i=torch.zeros(
            input_tensor.shape[1],
            layer.cell.hidden_size,
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        ),
    )
    out, _ = layer(input_tensor, state)

    assert out.shape == (10, 5, 4)


def test_lif_ex_feedforward_cell():
    layer = LIFExFeedForwardCell()
    data = torch.randn(5, 2, 4)
    out, _ = layer(data)

    assert out.shape == (5, 2, 4)


def test_lif_ex_feedforward_cell_state():
    layer = LIFExFeedForwardCell()
    input_tensor = torch.randn(5, 2, 4)
    state = LIFExFeedForwardState(
        v=layer.p.v_leak,
        i=torch.zeros(
            *input_tensor.shape, device=input_tensor.device, dtype=input_tensor.dtype
        ),
    )
    out, _ = layer(input_tensor, state)

    assert out.shape == (5, 2, 4)


def test_lif_ex_feedforward_cell_backward():
    layer = LIFExFeedForwardCell()
    data = torch.randn(5, 2, 4)
    out, _ = layer(data)
    out.sum().backward()


def test_lif_ex_feedforward_cell_repr():
    model = LIFExFeedForwardCell()
    assert (
        str(model)
        == "LIFExFeedForwardCell(p=LIFExParameters(delta_T=tensor(0.5000), tau_syn_inv=tensor(200.), tau_mem_inv=tensor(100.), v_leak=tensor(0.), v_th=tensor(1.), v_reset=tensor(0.), method='super', alpha=100.0), dt=0.001)"
    )


def test_lif_ex_cell_repr():
    model = LIFExCell(2, 4)
    assert (
        str(model)
        == "LIFExCell(2, 4, p=LIFExParameters(delta_T=tensor(0.5000), tau_syn_inv=tensor(200.), tau_mem_inv=tensor(100.), v_leak=tensor(0.), v_th=tensor(1.), v_reset=tensor(0.), method='super', alpha=100.0), dt=0.001)"
    )
