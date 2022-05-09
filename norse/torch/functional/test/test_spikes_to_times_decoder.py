import torch
from norse.torch.functional.spikes_to_times_decoder import SpikesToTimesDecoder


def test_zero_input():
    decoder_w_time = SpikesToTimesDecoder(
        spike_count=10, convert_indices_to_times=False
    )
    decoder_wo_time = SpikesToTimesDecoder(
        spike_count=10, convert_indices_to_times=True, dt=1e-3
    )
    assert torch.allclose(
        decoder_wo_time(torch.zeros(10, 1, 1)), torch.as_tensor(torch.inf)
    )
    assert torch.allclose(
        decoder_w_time(torch.zeros(10, 1, 1)), torch.as_tensor(torch.inf)
    )


def test_ones_input():
    decoder_wo_time = SpikesToTimesDecoder(
        spike_count=10, convert_indices_to_times=False
    )
    decoder_w_time = SpikesToTimesDecoder(
        spike_count=10, convert_indices_to_times=True, dt=1e-3
    )
    assert torch.allclose(
        decoder_wo_time(torch.ones(10, 1, 1)),
        torch.arange(10, dtype=torch.float)[:, None, None],
    )
    assert torch.allclose(
        decoder_w_time(torch.ones(10, 1, 1)),
        1e-3 * torch.arange(10, dtype=torch.float)[:, None, None],
    )


def test_single_spike_input_with_time():
    decoder = SpikesToTimesDecoder(
        spike_count=1, convert_indices_to_times=True, dt=1e-3
    )
    test_input = torch.zeros(10, 2, 3)
    test_input[2, :, :] = 1.0
    result = decoder(test_input)
    assert torch.allclose(result, torch.ones(1, 2, 3) * 2.0e-3)


def test_backward():
    # test backward without time conversion of spikes indices
    test_input = torch.zeros(10, 2, 3)
    test_input[1, :, :] = 1.0
    test_input.requires_grad_(True)

    decoder_w_time = SpikesToTimesDecoder(spike_count=1, convert_indices_to_times=False)
    result = decoder_w_time(test_input).sum()
    result.backward()
    expected_grad = torch.zeros_like(test_input)
    expected_grad[1, :, :] = -1.0
    assert torch.equal(test_input.grad, expected_grad)

    # test backward with time conversion using dt=1e-3
    test_input = torch.zeros(10, 2, 3)
    test_input[1, :, :] = 1.0
    test_input.requires_grad_(True)

    decoder_wo_time = SpikesToTimesDecoder(
        spike_count=1, convert_indices_to_times=True, dt=1e-3
    )
    result = decoder_wo_time(test_input).sum()
    result.backward()
    expected_grad[1, :, :] = -decoder_wo_time.dt
    assert torch.equal(test_input.grad, expected_grad)
