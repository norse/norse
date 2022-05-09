import torch


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
        self.decoding_fn = self.ToSpikeTimes.apply

    def forward(self, spike_input):
        """
        Call decoder custom autograd function

        :param spike_input: spike tensor with 0s and 1s (indicating spikes)
        """
        spike_indices = self.decoding_fn(spike_input, self.spike_count)
        if self.convert_to_time:
            spike_indices = spike_indices * self.dt
        return spike_indices

    @staticmethod
    class ToSpikeTimes(torch.autograd.Function):
        """
        Convert spike input with shape [time_steps x batch_size x output_size] to the indices of
        spike times, shape [spike_count x batch_size x output_size]. If no spike is present, time
        or index is set to inf.
        """

        @staticmethod
        def forward(ctx, spike_input, spike_count):
            """
            Return indices of first spike_count spikes (if spike_count < spike_input.shape[0],
            i.e. its time dim). If less than spike_count spikes happened along a output trace,
            its remaining entries are set to inf.
            """
            indexed_spike_input = (
                spike_input * torch.arange(1, spike_input.shape[0] + 1)[:, None, None]
                - 1.0
            )
            indexed_spike_input[indexed_spike_input == -1.0] = torch.inf
            spike_indices = torch.sort(indexed_spike_input, dim=0).values[
                : spike_count
                if spike_count < spike_input.shape[0]
                else spike_input.shape[0]
            ]
            ctx.save_for_backward(spike_indices, spike_input)
            ctx.shape = spike_input.shape
            return spike_indices

        @staticmethod
        def backward(ctx, grad_output):
            """
            Local gradient is set -1 for spike whose index was returned by forward, and 0 for no
            spike or if a spikes index wasn't returned in forward call.
            """
            (spike_indices, spk_rec) = ctx.saved_tensors
            spikeidcs_size, batch_size, out_size = spike_indices.shape
            noninf_spike_indices = spike_indices.flatten() != torch.inf

            grad_local = torch.zeros_like(spk_rec, dtype=torch.float)

            grad_local_indices = (
                spike_indices.flatten().type(torch.int64)[noninf_spike_indices],
                torch.arange(batch_size)
                .repeat_interleave(out_size)
                .repeat(spikeidcs_size)[noninf_spike_indices],
                torch.arange(out_size)
                .repeat(batch_size)
                .repeat(spikeidcs_size)[noninf_spike_indices],
            )

            grad_local[grad_local_indices] = 1.0

            grad_input = -1.0 * grad_output * grad_local

            return grad_input, None
