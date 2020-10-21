import torch

from norse.torch.module.coba_lif import CobaLIFCell, CobaLIFState


def test_coba():
    cell = CobaLIFCell(4, 3)
    data = torch.ones(5, 4)
    spikes, state = cell(data)
    assert spikes.shape == (5, 3)
    assert state.v.shape == (5, 3)


def test_coba_state():
    cell = CobaLIFCell(4, 3)
    input_tensor = torch.ones(5, 4)
    state = CobaLIFState(
        z=torch.zeros(
            input_tensor.shape[0],
            cell.hidden_size,
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        ),
        v=torch.zeros(
            input_tensor.shape[0],
            cell.hidden_size,
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        ),
        g_e=torch.zeros(
            input_tensor.shape[0],
            cell.hidden_size,
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        ),
        g_i=torch.zeros(
            input_tensor.shape[0],
            cell.hidden_size,
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        ),
    )

    spikes, state = cell(input_tensor, state)
    assert spikes.shape == (5, 3)
    assert state.v.shape == (5, 3)


def test_coba_backward():
    cell = CobaLIFCell(4, 3)
    data = torch.ones(5, 4)
    spikes, _ = cell(data)
    spikes.sum().backward()
