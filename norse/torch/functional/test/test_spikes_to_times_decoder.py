import torch
from norse.torch.functional.spikes_to_times_decoder import ToSpikeTimes


def test_zero_input():
    assert torch.allclose(
        ToSpikeTimes.apply(torch.zeros(10, 1, 1), torch.as_tensor(10)),
        torch.as_tensor(torch.inf),
    )


def test_ones_input():
    assert torch.allclose(
        ToSpikeTimes.apply(torch.ones(10, 1, 1), torch.as_tensor(10)),
        torch.arange(10, dtype=torch.float)[:, None, None],
    )


def test_single_spike_input_with_time():
    test_input = torch.zeros(10, 2, 3)
    test_input[2, :, :] = 1.0
    result = ToSpikeTimes.apply(test_input, torch.as_tensor(1))
    assert torch.allclose(result, torch.ones(1, 2, 3) * 2.0)


def test_backward():
    # test backward without time conversion of spikes indices
    test_input = torch.zeros(10, 2, 3)
    test_input[1, :, :] = 1.0
    test_input.requires_grad_(True)

    result = ToSpikeTimes.apply(test_input, torch.as_tensor(1)).sum()
    result.backward()
    expected_grad = torch.zeros_like(test_input)
    expected_grad[1, :, :] = -1.0
    assert torch.equal(test_input.grad, expected_grad)
