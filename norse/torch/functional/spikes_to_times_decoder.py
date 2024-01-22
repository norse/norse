import torch


class ToSpikeTimes(torch.autograd.Function):
    """Convert spike input with shape [time_steps x batch_size x output_size]
    to the indices of spike times, shape [spike_count x batch_size x
    output_size]. If no spike is present, time or index is set to inf.
    """

    @staticmethod
    def forward(ctx, spike_input: torch.Tensor, spike_count: torch.Tensor):
        """Return indices of first spike_count spikes (if spike_count < spike_input.shape[0],
            i.e. its time dim). If less than spike_count spikes happened along a output trace,
            its remaining entries are set to inf.

        Parameters:
            spike_input: spike tensor with 0s and 1s (indicating spikes)
            spike_count: number of elements (ascending) to return for each output channel.
        """
        indexed_spike_input = (
            spike_input * torch.arange(1, spike_input.shape[0] + 1)[:, None, None] - 1.0
        )
        indexed_spike_input[indexed_spike_input == -1.0] = torch.inf
        spike_indices = torch.sort(indexed_spike_input, dim=0).values[
            : (
                spike_count
                if spike_count < spike_input.shape[0]
                else spike_input.shape[0]
            )
        ]
        ctx.save_for_backward(spike_indices, spike_input)
        ctx.shape = spike_input.shape
        return spike_indices

    @staticmethod
    def backward(ctx, grad_output):
        """Local gradient is set -1 for spike whose index was returned by forward, and 0 for no
        spike or if a spikes index wasn't returned in forward call.
        """
        (spike_indices, spk_rec) = ctx.saved_tensors
        spikeidcs_size, batch_size, out_size = spike_indices.shape
        noninf_spike_indices = spike_indices.flatten() != torch.inf

        grad_input = torch.zeros_like(spk_rec, dtype=torch.float)

        grad_input_indices = (
            spike_indices.flatten().type(torch.long)[noninf_spike_indices],
            torch.arange(batch_size)
            .repeat_interleave(out_size)
            .repeat(spikeidcs_size)[noninf_spike_indices],
            torch.arange(out_size)
            .repeat(batch_size)
            .repeat(spikeidcs_size)[noninf_spike_indices],
        )

        grad_input[grad_input_indices] = -1.0 * grad_output.flatten()

        return grad_input, None
