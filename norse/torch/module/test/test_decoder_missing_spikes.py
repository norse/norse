import pytest
import torch

from norse.torch.module.spikes_to_times_decoder import SpikesToTimesDecoder


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("pattern", ["full", "mixed", "silent"])
@pytest.mark.parametrize("spike_count", [1, 3])
@pytest.mark.parametrize("convert_to_time", [False, True])
def test_decoder_backward_with_missing_spikes(
    dtype, pattern, spike_count, convert_to_time
):
    spikes = torch.zeros((5, 2, 3), dtype=dtype)
    if pattern == "full":
        events = [[0, 2, 4]] * 6
    elif pattern == "mixed":
        events = [[0, 3], [], [2], [1], [0, 2, 4], []]
    else:
        events = [[]] * 6
    for channel, times in enumerate(events):
        for time in times:
            spikes[time, channel // 3, channel % 3] = 1
    spikes.requires_grad_()
    scale = 0.25 if convert_to_time else 1.0
    weights = (
        torch.arange(1, spike_count * 6 + 1, dtype=dtype).reshape(spike_count, 2, 3)
        / 32
    )
    expected_times = torch.full((spike_count, 2, 3), torch.inf, dtype=dtype)
    expected_grad = torch.zeros_like(spikes)
    for channel, times in enumerate(events):
        batch, neuron = channel // 3, channel % 3
        for rank, time in enumerate(times[:spike_count]):
            expected_times[rank, batch, neuron] = time * scale
            expected_grad[time, batch, neuron] = -weights[rank, batch, neuron] * scale

    decoder = SpikesToTimesDecoder(
        spike_count=spike_count, convert_indices_to_times=convert_to_time, dt=0.25
    )
    actual = decoder(spikes)
    torch.testing.assert_close(actual, expected_times)
    loss = (torch.where(torch.isfinite(actual), actual, 0) * weights).sum()
    loss.backward()

    torch.testing.assert_close(spikes.grad, expected_grad)
