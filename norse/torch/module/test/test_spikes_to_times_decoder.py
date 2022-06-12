import torch
from norse.torch.module.spikes_to_times_decoder import SpikesToTimesDecoder


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

    decoder_wo_time = SpikesToTimesDecoder(
        spike_count=1, convert_indices_to_times=False
    )
    result = decoder_wo_time(test_input).sum()
    result.backward()
    expected_grad = torch.zeros_like(test_input)
    expected_grad[1, :, :] = -1.0
    assert torch.equal(test_input.grad, expected_grad)

    # test backward with time conversion using dt=1e-3
    test_input = torch.zeros(10, 2, 3)
    test_input[1, :, :] = 1.0
    test_input.requires_grad_(True)

    decoder_w_time = SpikesToTimesDecoder(
        spike_count=1, convert_indices_to_times=True, dt=1e-3
    )
    result = decoder_w_time(test_input).sum()
    result.backward()
    expected_grad[1, :, :] = -decoder_w_time.dt
    assert torch.equal(test_input.grad, expected_grad)

    # test backward for 1 < spike_count < test_input.shape[0]
    for spike_count in range(5):
        test_input = torch.ones(5, 3, 2).requires_grad_(True)
        decoder = SpikesToTimesDecoder(
            spike_count=spike_count, convert_indices_to_times=False
        )
        spike_times = decoder(test_input)
        scale_parameters = torch.arange(spike_times.flatten().shape[0]).reshape(
            spike_times.shape
        )
        result = torch.sum(spike_times * scale_parameters)
        result.backward()
        expected_grad = torch.zeros_like(test_input)
        expected_grad[:spike_count, :, :] = -scale_parameters
        assert torch.allclose(test_input.grad, expected_grad)


def test_jit():
    test_input = torch.zeros((2, 1, 1))
    test_input[1, :, :] = 1.0

    traced_decoder_module = torch.jit.trace_module(
        SpikesToTimesDecoder(
            spike_count=torch.as_tensor(2), convert_indices_to_times=True, dt=1e-3
        ),
        inputs={"forward": torch.zeros(1, 1, 1)},
    )

    assert torch.equal(
        traced_decoder_module(test_input), torch.tensor([[[1e-3]], [[torch.inf]]])
    )
