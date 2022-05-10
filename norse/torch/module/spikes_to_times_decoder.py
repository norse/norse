import torch

from norse.torch.functional.spikes_to_times_decoder import ToSpikeTimes


class SpikesToTimesDecoder(torch.nn.Module):
    """
    Module wrapper for spike to spike-time (or spike-index) decoder.

    Decodes input spikes with shape [time_steps x batch_size x output_size] into spike indices
    tensor with shape [spike_count x batch_size x output_size] in ascending order, e.g. for
    spike_count = 1 returns only times of first spikes along each batch- and output-dimension.
    If stated, converts indices along time axis into times with given dt value.
    """

    def __init__(
        self,
        spike_count: int = 1,
        convert_indices_to_times: bool = True,
        dt: float = 1e-3,
    ):
        """
        Initialize decoder.
        :param spike_count: number of elements (ascending) to return for each output channel.
        :param convert_indices_to_times: Whether to return times or indices along input's time axis
        :param dt: time step of input's time axis (needed for conversion of indices to times)
        """
        super().__init__()
        self.spike_count = spike_count
        self.convert_to_time = convert_indices_to_times
        self.dt = dt
        self.decoding_fn = ToSpikeTimes.apply

    def forward(self, spike_input):
        """
        Call decoder custom autograd function

        :param spike_input: spike tensor with 0s and 1s (indicating spikes)
        """
        spike_indices = self.decoding_fn(spike_input, self.spike_count)
        if self.convert_to_time:
            spike_indices = spike_indices * self.dt
        return spike_indices
