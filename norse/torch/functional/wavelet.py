r"""Time-causal and scale-covariant wavelets.

Implements the spiking bandpass wavelets of Pedersen, Lindeberg and
Gerstoft, *Encoding and Decoding Temporal Signals with Spiking Bandpass
Wavelets*, arXiv:2605.09770 (2026), ported from the Jax reference
implementation at https://github.com/jegp/swavelet.

The analysis stage is a bank of cascaded truncated exponentials. A single
stage integrates

.. math::
    \dot L = \frac{1}{\mu}\left(f - L\right),

whose exact discretisation at step :math:`\Delta t` is

.. math::
    L[n] = \alpha L[n-1] + (1 - \alpha) f[n],
    \qquad \alpha = e^{-\Delta t / \mu}.

Two equivalent evaluation paths are provided, because they have very
different costs:

* **Kernel path** (:func:`time_causal_limit_kernel`,
  :func:`dot_analysis`): the cascade's impulse response is built once and
  applied by FFT convolution, costing :math:`O(T \log T)` for the whole
  signal. Use this whenever the full signal is available.
* **Recursive path** (:func:`exponential_tau_mem_inv`): the same filter
  expressed as parameters for :class:`norse.torch.LIBoxCell`, stepped one
  sample at a time. Needed for the spiking variants, where the firing
  threshold makes the recursion genuinely sequential, and for NIR export.

The recursive path uses Norse's leaky integrators as they are. They
advance by ``v <- v + dt * tau_mem_inv * ((v_leak - v) + i)``, so setting
``dt * tau_mem_inv = 1 - alpha`` gives the exponential update above; see
:func:`norse.torch.functional.scale_space.exponential_tau_mem_inv`.
"""

import math
from typing import Optional, Sequence

import torch

from norse.torch.functional.scale_space import (  # noqa: F401
    exponential_tau_mem_inv,
    geometric_time_constants,
    per_step_leak,
    stable_cascade_depth,
)


def time_causal_limit_kernel(
    stage_taus: Sequence[float],
    dt: float,
    length: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    r"""Impulse response of a cascade of truncated exponentials.

    Approximates Lindeberg's time-causal limit kernel
    :math:`\Psi(t; \mu, c)` by composing one first-order recursion per
    stage. Each stage has impulse response
    :math:`h[n] = (1-\alpha)\alpha^n`, and the cascade is their
    convolution, computed here by running the recursions over an impulse.

    Args:
        stage_taus: Per-stage time constants in seconds, e.g. from
            :func:`geometric_time_constants`.
        dt: Sample period in seconds.
        length: Number of samples of impulse response to return.
        dtype: Output dtype.
        device: Output device.

    Returns:
        Tensor of shape ``(length,)``, the causal impulse response,
        normalised so that it sums to approximately one (exactly one in the
        limit ``length -> inf``).
    """
    # Each stage's impulse response is available in closed form,
    #     h[n] = (1 - alpha) * alpha^n,
    # so the cascade is their convolution -- taken here as a product of
    # spectra rather than by running the recursions sample by sample.
    n_fft = 1 << (length * 2).bit_length()
    idx = torch.arange(length, dtype=torch.float64, device=device)
    spectrum = None
    for tau in stage_taus:
        alpha = math.exp(-dt / float(tau))
        h = (1.0 - alpha) * alpha**idx
        h_f = torch.fft.rfft(h, n_fft)
        spectrum = h_f if spectrum is None else spectrum * h_f
    if spectrum is None:
        raise ValueError("stage_taus must contain at least one time constant")
    return torch.fft.irfft(spectrum, n_fft)[:length].to(dtype)


def fft_convolve_causal(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    r"""Causal convolution of ``signal`` with ``kernel`` via FFT.

    Returns the first ``signal.shape[-1]`` samples of the full
    convolution, so output sample ``n`` depends only on input samples
    ``<= n``. Batched over all leading dimensions of ``signal``; ``kernel``
    may carry one leading channel dimension, which is broadcast against
    ``signal``.

    Args:
        signal: Shape ``(..., T)``.
        kernel: Shape ``(T_k,)`` or ``(C, T_k)``.

    Returns:
        Shape ``(..., T)`` if ``kernel`` is 1-D, else ``(C, ..., T)``.
    """
    t = signal.shape[-1]
    n = 1 << (t + kernel.shape[-1] - 1).bit_length()
    sig_f = torch.fft.rfft(signal, n, dim=-1)
    ker_f = torch.fft.rfft(kernel, n, dim=-1)
    if kernel.dim() == 2:
        sig_f = sig_f.unsqueeze(0)
        ker_f = ker_f.reshape(kernel.shape[0], *([1] * (signal.dim() - 1)), -1)
    return torch.fft.irfft(sig_f * ker_f, n, dim=-1)[..., :t]


def dot_analysis(signal: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    r"""Bandpass differences plus lowpass residual, shape ``(K+1, ..., T)``.

    With smoothing scales :math:`\mu_1 < \dots < \mu_K` supplied as
    ``kernels``, and :math:`L_k = \Psi(\cdot;\mu_k) * f`, the first
    :math:`K` output channels are the bandpass differences

    .. math::
        \Delta L_1 = L_1 - f, \qquad
        \Delta L_k = L_k - L_{k-1}, \quad k = 2 \dots K,

    taking the raw signal as the implicit finest level :math:`L_0 = f`, and
    the last channel is the lowpass residual :math:`L_K`. This is what
    makes :func:`dot_reconstruct` a telescoping sum and so exactly
    invertible.

    Args:
        signal: Shape ``(..., T)``.
        kernels: Shape ``(K, T_k)``, ordered finest-to-coarsest scale.

    Returns:
        Shape ``(K + 1, ..., T)``.
    """
    scales = fft_convolve_causal(signal, kernels)
    bandpass = torch.cat(
        [(scales[0] - signal).unsqueeze(0), scales[1:] - scales[:-1]], dim=0
    )
    return torch.cat([bandpass, scales[-1:]], dim=0)


def dot_reconstruct(
    per_channel: torch.Tensor, channel_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""Invert :func:`dot_analysis` by telescoping the channels.

    .. math::
        f = w_K x_K - \sum_{k=1}^{K} w_k x_k,

    where :math:`x_K` is the lowpass residual and :math:`x_1 \dots x_K` the
    bandpass differences. Unit weights give exact reconstruction; fitted
    weights compensate for truncating the coarsest scale at finite
    :math:`\mu`.

    Args:
        per_channel: Shape ``(K + 1, ..., T)`` as returned by
            :func:`dot_analysis`.
        channel_weights: Shape ``(K + 1,)``. Defaults to all ones.

    Returns:
        Shape ``(..., T)``.
    """
    if channel_weights is None:
        return per_channel[-1] - per_channel[:-1].sum(dim=0)
    w = channel_weights.reshape(-1, *([1] * (per_channel.dim() - 1)))
    return w[-1] * per_channel[-1] - (w[:-1] * per_channel[:-1]).sum(dim=0)


def bandpass_kernel_norms(kernels: torch.Tensor) -> torch.Tensor:
    r"""L2 norms of the analysis channels' effective kernels.

    Channel :math:`k` of :func:`dot_analysis` applies
    :math:`\Psi(\mu_1) - \delta` for :math:`k = 0`,
    :math:`\Psi(\mu_{k+1}) - \Psi(\mu_k)` for :math:`1 \le k < K`, and
    :math:`\Psi(\mu_K)` for the lowpass residual. Dividing each channel by
    its norm equalises the channels' response to white input, which is what
    gives a single spiking threshold comparable meaning across scales.

    Args:
        kernels: Shape ``(K, T)``, the smoothing cascades' impulse responses.

    Returns:
        Shape ``(K + 1,)``, one norm per analysis channel.
    """
    delta = torch.zeros_like(kernels[0])
    delta[0] = 1.0
    bandpass = torch.cat(
        [(kernels[0] - delta).unsqueeze(0), kernels[1:] - kernels[:-1]], dim=0
    )
    channels = torch.cat([bandpass, kernels[-1:]], dim=0)
    return channels.pow(2).sum(dim=-1).sqrt()


def invert_membrane_integration(
    signed_spikes: torch.Tensor,
    threshold: torch.Tensor,
    mu_mem: torch.Tensor,
    dt: float,
    smooth: bool = True,
) -> torch.Tensor:
    r"""Recover channel coefficients from signed spike trains.

    The encoder drives each channel's membrane with the channel's analysis
    output :math:`x`,

    .. math:: v[n] = \alpha v[n-1] + (1 - \alpha) x[n],
              \qquad \alpha = e^{-\Delta t/\mu_{\text{mem}}},

    and emits a spike whenever :math:`v` exceeds :math:`\theta`, resetting to
    zero. Each spike therefore accounts for :math:`\theta` of membrane
    charge, which arrived as :math:`(1 - \alpha) x`, so

    .. math:: \hat x = \frac{\theta}{1 - \alpha}\,
                       \left(h_{\text{mem}} * s\right),

    with :math:`s` the signed train and :math:`h_{\text{mem}}` the membrane
    response, which turns the spike train back into the continuous quantity
    the threshold was applied to.

    The estimate is biased low: a spike discharges the membrane from
    :math:`v \ge \theta`, not from exactly :math:`\theta`, so the charge per
    spike exceeds :math:`\theta` whenever the drive is large compared with
    the threshold. Fitting one weight per channel removes the bias; see
    :meth:`~norse.torch.module.wavelet.SpikingDifferenceOfTimeCausalKernels.fit_channel_weights`.

    Args:
        signed_spikes: Shape ``(C, ..., T)``, the positive train minus the
            negative train for each channel.
        threshold: Shape ``(C,)``, per-channel firing threshold.
        mu_mem: Shape ``(C,)``, per-channel membrane time constant.
        dt: Sample period in seconds.
        smooth: Convolve with the membrane response. With ``False`` the
            spikes are only rescaled, giving an impulsive estimate.

    Returns:
        Shape ``(C, ..., T)``, the recovered coefficients.
    """
    length = signed_spikes.shape[-1]
    dtype, device = signed_spikes.dtype, signed_spikes.device
    alpha = torch.exp(-dt / mu_mem.to(dtype=dtype, device=device))
    gain = (threshold.to(dtype=dtype, device=device) / (1 - alpha)).reshape(
        -1, *([1] * (signed_spikes.dim() - 1))
    )
    if not smooth:
        return gain * signed_spikes

    idx = torch.arange(length, dtype=dtype, device=device)
    membrane = (1 - alpha).unsqueeze(-1) * alpha.unsqueeze(-1) ** idx
    n = 1 << (2 * length).bit_length()
    spikes_f = torch.fft.rfft(signed_spikes, n, dim=-1)
    membrane_f = torch.fft.rfft(membrane, n, dim=-1).reshape(
        membrane.shape[0], *([1] * (signed_spikes.dim() - 2)), -1
    )
    smoothed = torch.fft.irfft(spikes_f * membrane_f, n, dim=-1)[..., :length]
    return gain * smoothed
