r"""
Transmission of a spike from one neuron to another takes a real, non-zero
delay (the axonal / synaptic propagation time), which depends on the
distance between neurons and the axon type. This module provides that as a
first-class, differentiable primitive, so a delay can be learned by
backprop rather than fixed by hand -- see issue #224 and, for the specific
learnable-delay formulation used here, Hammouamri, Khalfaoui-Hassani &
Masquelier, *Learning Delays in Spiking Neural Networks using Dilated
Convolutions with Learnable Spacings*, ICLR 2024.

The delay is realised by interpolation over a ring buffer of the most
recent ``max_delay`` inputs, so a delay of ``d`` steps need not be an
integer: gradients flow to ``d`` itself.
"""

from typing import NamedTuple, Optional, Tuple

import torch


class SynapseDelayState(NamedTuple):
    """State of a :class:`SynapseDelay`: a ring buffer of the most recent
    ``max_delay + 1`` inputs, most-recent first.

    Parameters:
        buffer (torch.Tensor): shape ``(max_delay + 1, *input_shape)``
    """

    buffer: torch.Tensor


def _delay_weights(
    delay: torch.Tensor,
    max_delay: int,
    kernel: str,
    sigma: float,
    device,
    dtype,
) -> torch.Tensor:
    """Per-lag interpolation weights, shape ``(max_delay + 1, *delay.shape)``.
    ``weights[l, ...]`` is the fraction of the value from ``l`` steps ago
    that reaches the output, for a delay of ``delay[...]`` steps."""
    d = delay.to(device=device, dtype=dtype).clamp(0.0, max_delay)
    lags = torch.arange(max_delay + 1, device=device, dtype=dtype)
    lags = lags.reshape(-1, *([1] * d.dim()))  # (K+1, 1, 1, ...)
    diff = lags - d.unsqueeze(0)

    if kernel == "linear":
        # exact 2-tap kernel: w[floor(d)] = 1-frac(d), w[floor(d)+1] = frac(d)
        weights = (1.0 - diff.abs()).clamp(min=0.0)
    elif kernel == "gaussian":
        # wider gradient support; anneal `sigma` from wide to narrow to
        # learn a large delay from a far initial value (see module docs)
        weights = torch.exp(-(diff**2) / (2.0 * sigma**2))
    else:
        raise ValueError(f"kernel must be 'linear' or 'gaussian', got {kernel!r}")

    return weights / weights.sum(dim=0, keepdim=True).clamp(min=1e-12)


def _broadcast_to(weights: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Insert size-1 dims after dim 0 until `weights` broadcasts against a
    `target_dim`-dimensional buffer whose trailing dims match `delay`'s."""
    while weights.dim() < target_dim:
        weights = weights.unsqueeze(1)
    return weights


def synapse_delay_step(
    input_tensor: torch.Tensor,
    state: Optional[SynapseDelayState],
    delay: torch.Tensor,
    max_delay: int,
    kernel: str = "linear",
    sigma: float = 1.0,
) -> Tuple[torch.Tensor, SynapseDelayState]:
    """Advance the delay line by one step.

    Parameters:
        input_tensor (torch.Tensor): this step's input
        state (Optional[SynapseDelayState]): the ring buffer, or None to
            start from all-zero history
        delay (torch.Tensor): scalar, or a tensor broadcastable against
            ``input_tensor`` (e.g. one delay per feature channel)
        max_delay (int): ring-buffer / kernel length
        kernel (str): ``"linear"`` (default) or ``"gaussian"``
        sigma (float): Gaussian kernel width (ignored for ``"linear"``)

    Returns:
        A tuple of (delayed output, new :class:`SynapseDelayState`).
    """
    if state is None or state.buffer.shape[1:] != input_tensor.shape:
        buffer = torch.zeros(
            max_delay + 1,
            *input_tensor.shape,
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )
    else:
        buffer = state.buffer
    buffer = torch.cat((input_tensor.unsqueeze(0), buffer[:-1]), dim=0)

    weights = _delay_weights(
        delay, max_delay, kernel, sigma, input_tensor.device, input_tensor.dtype
    )
    weights = _broadcast_to(weights, buffer.dim())
    output = (buffer * weights).sum(dim=0)

    return output, SynapseDelayState(buffer=buffer)
