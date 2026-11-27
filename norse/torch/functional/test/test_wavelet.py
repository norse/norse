import math

import numpy as np
import pytest
import torch

from norse.torch.functional.scale_space import geometric_time_constants
from norse.torch.functional.wavelet import (
    bandpass_kernel_norms,
    dot_analysis,
    invert_membrane_integration,
    dot_reconstruct,
    fft_convolve_causal,
    time_causal_limit_kernel,
)


def _reference_cascade(stage_taus, dt, length):
    """Impulse response by running each stage's recursion sample by sample."""
    u = np.zeros(length)
    u[0] = 1.0
    for tau in stage_taus:
        alpha = math.exp(-dt / float(tau))
        out, acc = np.empty(length), 0.0
        for n in range(length):
            acc = alpha * acc + (1 - alpha) * u[n]
            out[n] = acc
        u = out
    return u


def _kernels(mus, dt, c, n_stages, length, dtype=torch.float64):
    return torch.stack(
        [
            time_causal_limit_kernel(
                geometric_time_constants(mu, dt, c, n_stages).tolist(),
                dt,
                length,
                dtype=dtype,
            )
            for mu in mus
        ]
    )


def test_limit_kernel_matches_sequential_cascade():
    dt, length = 1e-3, 512
    taus = geometric_time_constants(0.05, dt, 2.0, 4).tolist()
    got = time_causal_limit_kernel(taus, dt, length, dtype=torch.float64).numpy()
    assert np.allclose(got, _reference_cascade(taus, dt, length), atol=1e-14)


def test_limit_kernel_is_normalised():
    dt = 1e-3
    taus = geometric_time_constants(0.01, dt, 2.0, 3).tolist()
    kernel = time_causal_limit_kernel(taus, dt, 4096, dtype=torch.float64)
    assert kernel.sum().item() == pytest.approx(1.0, abs=1e-9)


def test_limit_kernel_is_causal_and_non_negative():
    dt = 1e-3
    taus = geometric_time_constants(0.02, dt, 2.0, 4).tolist()
    kernel = time_causal_limit_kernel(taus, dt, 1024, dtype=torch.float64)
    assert torch.all(kernel >= -1e-15)
    # A cascade of causal filters starts at zero and rises to a single peak.
    assert kernel[0].item() < kernel.max().item()
    assert kernel.argmax().item() > 0


def test_limit_kernel_single_stage_is_the_exponential():
    dt, mu, length = 1e-3, 5e-3, 256
    kernel = time_causal_limit_kernel([mu], dt, length, dtype=torch.float64)
    alpha = math.exp(-dt / mu)
    expected = (1 - alpha) * alpha ** torch.arange(length, dtype=torch.float64)
    assert torch.allclose(kernel, expected, atol=1e-14)


def test_limit_kernel_rejects_empty_cascade():
    with pytest.raises(ValueError, match="at least one time constant"):
        time_causal_limit_kernel([], 1e-3, 16)


def test_fft_convolve_causal_is_causal():
    # An impulse at t=k cannot influence outputs before k.
    signal = torch.zeros(64, dtype=torch.float64)
    signal[20] = 1.0
    kernel = torch.ones(8, dtype=torch.float64) / 8
    out = fft_convolve_causal(signal, kernel)
    assert torch.allclose(out[:20], torch.zeros(20, dtype=torch.float64), atol=1e-12)
    assert out[20].item() == pytest.approx(0.125)


def test_fft_convolve_causal_batches_channels():
    signal = torch.randn(3, 5, 64, dtype=torch.float64)
    kernels = torch.randn(4, 16, dtype=torch.float64)
    out = fft_convolve_causal(signal, kernels)
    assert out.shape == (4, 3, 5, 64)
    single = fft_convolve_causal(signal, kernels[2])
    assert torch.allclose(out[2], single, atol=1e-12)


def test_dot_analysis_shape_and_channel_meaning():
    dt, length = 1e-3, 1024
    mus = [0.0125, 0.025, 0.05]
    kernels = _kernels(mus, dt, 2.0, 4, length)
    signal = torch.randn(length, dtype=torch.float64)
    coeffs = dot_analysis(signal, kernels)

    assert coeffs.shape == (len(mus) + 1, length)
    smoothed = fft_convolve_causal(signal, kernels)
    # channel 0 is L_1 - f; channel k is L_{k+1} - L_k; last is L_K
    assert torch.allclose(coeffs[0], smoothed[0] - signal, atol=1e-12)
    assert torch.allclose(coeffs[1], smoothed[1] - smoothed[0], atol=1e-12)
    assert torch.allclose(coeffs[-1], smoothed[-1], atol=1e-12)


def test_dot_reconstruction_is_exact():
    dt, length = 1e-3, 2048
    kernels = _kernels([0.0125, 0.025, 0.05], dt, 2.0, 4, length)
    signal = torch.randn(length, dtype=torch.float64)
    recon = dot_reconstruct(dot_analysis(signal, kernels))
    assert torch.allclose(recon, signal, atol=1e-12)


def test_dot_reconstruction_exact_for_batched_signals():
    dt, length = 1e-3, 512
    kernels = _kernels([0.01, 0.02], dt, 2.0, 3, length)
    signal = torch.randn(2, 7, length, dtype=torch.float64)
    recon = dot_reconstruct(dot_analysis(signal, kernels))
    assert recon.shape == signal.shape
    assert torch.allclose(recon, signal, atol=1e-12)


def test_dot_reconstruct_honours_channel_weights():
    coeffs = torch.randn(4, 32, dtype=torch.float64)
    weights = torch.tensor([0.5, 2.0, -1.0, 3.0], dtype=torch.float64)
    expected = weights[-1] * coeffs[-1] - (weights[:-1, None] * coeffs[:-1]).sum(dim=0)
    assert torch.allclose(dot_reconstruct(coeffs, weights), expected, atol=1e-14)
    # Unit weights reproduce the default telescoping sum.
    unit = torch.ones(4, dtype=torch.float64)
    assert torch.allclose(
        dot_reconstruct(coeffs, unit), dot_reconstruct(coeffs), atol=1e-14
    )


def test_single_bandpass_channel_is_admissible():
    # A wavelet must have zero mean. The bandpass channels are differences of
    # normalised lowpass kernels, so their kernels integrate to zero.
    dt, length = 1e-3, 8192
    kernels = _kernels([0.01, 0.02], dt, 2.0, 3, length)
    impulse = torch.zeros(length, dtype=torch.float64)
    impulse[0] = 1.0
    coeffs = dot_analysis(impulse, kernels)
    assert coeffs[0].sum().item() == pytest.approx(0.0, abs=1e-6)
    assert coeffs[1].sum().item() == pytest.approx(0.0, abs=1e-6)


def test_invert_membrane_integration_scales_by_charge_per_spike():
    dt, mu_mem, threshold = 1e-3, 1e-3, 0.5
    spikes = torch.zeros(1, 64)
    spikes[0, 10] = 1.0
    out = invert_membrane_integration(
        spikes,
        torch.tensor([threshold]),
        torch.tensor([mu_mem]),
        dt,
        smooth=False,
    )
    alpha = math.exp(-dt / mu_mem)
    assert out[0, 10].item() == pytest.approx(threshold / (1 - alpha))
    assert out[0, 9].item() == 0.0


def test_invert_membrane_integration_is_causal():
    out = invert_membrane_integration(
        torch.nn.functional.one_hot(torch.tensor([20]), 64).float(),
        torch.tensor([0.5]),
        torch.tensor([5e-3]),
        1e-3,
    )
    # The FFT path leaves round-off rather than exact zeros.
    assert torch.allclose(out[0, :20], torch.zeros(20), atol=1e-6)
    # At the spike instant the recovered value is the threshold itself.
    assert out[0, 20].item() == pytest.approx(0.5, rel=1e-5)
    assert torch.all(out[0, 21:30] < out[0, 20])


def test_invert_membrane_integration_is_linear_in_the_spikes():
    threshold, mu_mem, dt = torch.tensor([0.3]), torch.tensor([2e-3]), 1e-3
    a = torch.randn(1, 64).sign()
    b = torch.randn(1, 64).sign()
    fa = invert_membrane_integration(a, threshold, mu_mem, dt)
    fb = invert_membrane_integration(b, threshold, mu_mem, dt)
    fab = invert_membrane_integration(a + b, threshold, mu_mem, dt)
    assert torch.allclose(fab, fa + fb, atol=1e-5)


def test_invert_membrane_integration_negates_with_the_spikes():
    threshold, mu_mem, dt = torch.tensor([0.3, 0.3]), torch.tensor([2e-3, 2e-3]), 1e-3
    s = torch.randn(2, 64).sign()
    assert torch.allclose(
        invert_membrane_integration(-s, threshold, mu_mem, dt),
        -invert_membrane_integration(s, threshold, mu_mem, dt),
        atol=1e-6,
    )


def test_bandpass_kernel_norms_counts_the_delta_in_the_leading_channel():
    dt = 1e-3
    kernels = _kernels([0.01, 0.02], dt, 2.0, 1, 4096)
    norms = bandpass_kernel_norms(kernels)
    assert norms.shape == (3,)
    # Psi(mu_1) - delta is dominated by the delta, so its norm is near one.
    assert 0.9 < norms[0] < 1.1
    # Adjacent-scale differences are much smaller.
    assert norms[1] < 0.2
