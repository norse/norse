import math

import pytest
import torch

from norse.torch.module.wavelet import (
    DifferenceOfExponentials,
    DifferenceOfTimeCausalKernels,
    SpikingDifferenceOfExponentials,
    SpikingDifferenceOfTimeCausalKernels,
)


def test_dot_analysis_shape():
    w = DifferenceOfTimeCausalKernels(n_channels=5, dt=1e-3, mu_max=0.05)
    coeffs = w(torch.randn(512))
    assert coeffs.shape == (512, 5)


def test_dot_reconstruction_is_exact():
    w = DifferenceOfTimeCausalKernels(n_channels=4, dt=1e-3, mu_max=0.05)
    signal = torch.randn(1024, dtype=torch.float64)
    recon = w.reconstruct(w(signal))
    assert torch.allclose(recon, signal, atol=1e-10)


def test_doe_reconstruction_is_exact():
    w = DifferenceOfExponentials(n_channels=3, dt=1e-3, mu_max=0.05)
    signal = torch.randn(1024, dtype=torch.float64)
    recon = w.reconstruct(w(signal))
    assert torch.allclose(recon, signal, atol=1e-10)


def test_doe_has_single_stage_cascades():
    w = DifferenceOfExponentials(n_channels=4)
    assert w.cascade_depths == [1, 1, 1]
    assert all(taus.numel() == 1 for taus in w.cascade_taus)


def test_batched_signals_round_trip():
    w = DifferenceOfTimeCausalKernels(n_channels=3, dt=1e-3, mu_max=0.02)
    signal = torch.randn(256, 2, 6, dtype=torch.float64)
    coeffs = w(signal)
    assert coeffs.shape == (256, 2, 6, 3)
    assert torch.allclose(w.reconstruct(coeffs), signal, atol=1e-10)


def test_scales_form_a_geometric_ladder_ending_at_mu_max():
    mu_max, c, n = 0.05, 2.0, 5
    w = DifferenceOfTimeCausalKernels(n_channels=n, dt=1e-3, mu_max=mu_max, c=c)
    mus = w.time_scales()
    assert mus.numel() == n - 1
    assert mus[-1].item() == pytest.approx(mu_max)
    assert torch.allclose(mus[1:] / mus[:-1], torch.full((n - 2,), c))


def test_transform_is_causal():
    # Perturbing the input at time k must leave outputs before k untouched.
    w = DifferenceOfTimeCausalKernels(n_channels=4, dt=1e-3, mu_max=0.02)
    signal = torch.randn(256, dtype=torch.float64)
    k = 128
    perturbed = signal.clone()
    perturbed[k] += 5.0
    a, b = w(signal), w(perturbed)
    assert torch.allclose(a[:k], b[:k], atol=1e-12)
    assert not torch.allclose(a[k], b[k], atol=1e-6)


def test_bandpass_channels_have_zero_mean_response():
    w = DifferenceOfTimeCausalKernels(n_channels=4, dt=1e-3, mu_max=0.01)
    impulse = torch.zeros(8192, dtype=torch.float64)
    impulse[0] = 1.0
    coeffs = w(impulse)
    for k in range(w.n_channels - 1):
        assert coeffs[:, k].sum().item() == pytest.approx(0.0, abs=1e-5)


def test_lowpass_channel_preserves_mean():
    w = DifferenceOfTimeCausalKernels(n_channels=3, dt=1e-3, mu_max=0.005)
    signal = torch.ones(4096, dtype=torch.float64)
    coeffs = w(signal)
    # The coarsest lowpass settles on the constant it is fed.
    assert coeffs[-1, -1].item() == pytest.approx(1.0, abs=1e-6)


def test_kernels_are_cached_per_length():
    w = DifferenceOfTimeCausalKernels(n_channels=3, dt=1e-3, mu_max=0.01)
    first = w.kernels(128)
    assert w.kernels(128) is first
    assert w.kernels(256) is not first


def test_kernel_length_bounds_the_impulse_response():
    w = DifferenceOfTimeCausalKernels(
        n_channels=3, dt=1e-3, mu_max=0.01, kernel_length=64
    )
    assert w.kernels(64).shape == (2, 64)
    assert w(torch.randn(512)).shape == (512, 3)


def test_learnable_channel_weights_are_parameters():
    fixed = DifferenceOfTimeCausalKernels(n_channels=3)
    assert not isinstance(fixed.channel_weights, torch.nn.Parameter)
    assert "channel_weights" not in dict(fixed.named_parameters())

    learned = DifferenceOfTimeCausalKernels(n_channels=3, learn_channel_weights=True)
    assert "channel_weights" in dict(learned.named_parameters())


def test_learnable_weights_receive_gradient():
    w = DifferenceOfTimeCausalKernels(
        n_channels=3, dt=1e-3, mu_max=0.01, learn_channel_weights=True
    )
    signal = torch.randn(128)
    w.reconstruct(w(signal)).pow(2).sum().backward()
    assert w.channel_weights.grad is not None
    assert torch.any(w.channel_weights.grad != 0)


def test_integrator_parameters_decay_exactly():
    dt = 1e-3
    w = DifferenceOfExponentials(n_channels=4, dt=dt, mu_max=0.02)
    p = w.integrator_parameters()
    taus = torch.stack([t[0] for t in w.cascade_taus]).float()
    multiplier = 1 - dt * p.tau_mem_inv
    assert torch.allclose(multiplier, torch.exp(-dt / taus), atol=1e-6)
    assert torch.all(multiplier > 0)


def test_integrators_match_the_kernel_convolution():
    # Stepping a channel's cascade cells reproduces that channel's smoothing.
    dt, length = 1e-3, 400
    w = DifferenceOfTimeCausalKernels(n_channels=3, dt=dt, mu_max=0.01)
    signal = torch.randn(length)
    kernels = w.kernels(length)
    from norse.torch.functional.wavelet import fft_convolve_causal

    expected = fft_convolve_causal(signal, kernels)
    banks = w.integrators()
    for channel, cells in enumerate(banks):
        out, states = [], [None] * len(cells)
        for n in range(length):
            v = signal[n : n + 1]
            for i, cell in enumerate(cells):
                v, states[i] = cell(v, states[i])
            out.append(v)
        got = torch.cat(out)
        assert torch.allclose(got, expected[channel], atol=1e-4)


def test_rejects_too_few_channels():
    with pytest.raises(ValueError, match="n_channels must be >= 2"):
        DifferenceOfTimeCausalKernels(n_channels=1)


def test_rejects_non_expanding_scale_ratio():
    with pytest.raises(ValueError, match="c must be > 1"):
        DifferenceOfTimeCausalKernels(n_channels=3, c=1.0)


def test_repr_reports_the_configuration():
    w = DifferenceOfTimeCausalKernels(n_channels=3, dt=1e-3, mu_max=0.01)
    text = repr(w)
    assert "n_channels=3" in text
    assert "cascade_depths=" in text


def test_spiking_output_is_twice_the_channels():
    w = SpikingDifferenceOfExponentials(n_channels=3, dt=1e-3, mu_max=0.02)
    spikes = w(torch.randn(256))
    assert spikes.shape == (256, 6)


def test_spikes_are_binary():
    w = SpikingDifferenceOfExponentials(n_channels=3, dt=1e-3, mu_max=0.02)
    spikes = w(torch.randn(512))
    assert torch.all((spikes == 0) | (spikes == 1))


def test_polarity_pairs_never_fire_together():
    # One neuron sees +x and the other -x, so a single sample cannot push
    # both membranes above threshold.
    w = SpikingDifferenceOfExponentials(n_channels=3, dt=1e-3, mu_max=0.02)
    spikes = w(torch.randn(512) * 5)
    assert torch.all(spikes[..., 0::2] * spikes[..., 1::2] == 0)


def test_spiking_encoder_responds_to_input():
    w = SpikingDifferenceOfExponentials(
        n_channels=3, dt=1e-3, mu_max=0.02, threshold=torch.tensor(0.05)
    )
    t = torch.arange(0, 1.0, 1e-3)
    signal = torch.sin(2 * math.pi * 5 * t)
    assert w(signal).sum() > 0


def test_silent_input_produces_no_spikes():
    w = SpikingDifferenceOfExponentials(n_channels=3, dt=1e-3, mu_max=0.02)
    assert w(torch.zeros(256)).sum() == 0


def test_higher_threshold_fires_less():
    t = torch.arange(0, 1.0, 1e-3)
    signal = torch.sin(2 * math.pi * 5 * t)
    low = SpikingDifferenceOfExponentials(
        n_channels=3, dt=1e-3, mu_max=0.02, threshold=torch.tensor(0.02)
    )
    high = SpikingDifferenceOfExponentials(
        n_channels=3, dt=1e-3, mu_max=0.02, threshold=torch.tensor(0.5)
    )
    assert low(signal).sum() > high(signal).sum()


def test_spiking_encoder_is_causal():
    w = SpikingDifferenceOfExponentials(
        n_channels=3, dt=1e-3, mu_max=0.01, threshold=torch.tensor(0.05)
    )
    signal = torch.randn(256)
    k = 128
    perturbed = signal.clone()
    perturbed[k] += 10.0
    assert torch.equal(w(signal)[:k], w(perturbed)[:k])


def test_channel_prescale_equalises_kernel_norms():
    w = SpikingDifferenceOfTimeCausalKernels(n_channels=4, dt=1e-3, mu_max=0.05)
    from norse.torch.functional.wavelet import bandpass_kernel_norms

    norms = bandpass_kernel_norms(w.analysis.kernels(4096))
    assert torch.allclose(w.channel_prescale(4096) * norms, torch.ones(4), atol=1e-4)


def test_normalize_false_gives_unit_prescale():
    w = SpikingDifferenceOfExponentials(n_channels=3, normalize=False)
    assert torch.allclose(w.channel_prescale(), torch.ones(3))


def test_neurons_decay_exactly_and_reset_to_zero():
    dt = 1e-3
    w = SpikingDifferenceOfExponentials(n_channels=3, dt=dt, mu_max=0.02)
    p = w.neurons().p
    expected = torch.exp(-dt / w.mu_mem).repeat_interleave(2)
    assert torch.allclose(1 - dt * p.tau_mem_inv, expected, atol=1e-6)
    assert torch.allclose(p.v_reset, torch.zeros(6))
    assert torch.allclose(p.v_th, w.threshold.repeat_interleave(2))


def test_default_threshold_scales_with_the_channel_scale():
    w = SpikingDifferenceOfTimeCausalKernels(n_channels=4, dt=1e-3, mu_max=0.05)
    # Coarser channels integrate more, so their default threshold is higher.
    assert torch.all(w.threshold[1:-1] > w.threshold[:-2])


def test_learnable_threshold_is_a_parameter():
    fixed = SpikingDifferenceOfExponentials(n_channels=3)
    assert "threshold" not in dict(fixed.named_parameters())
    learned = SpikingDifferenceOfExponentials(n_channels=3, learn_threshold=True)
    assert "threshold" in dict(learned.named_parameters())


def test_spiking_doe_is_depth_one():
    w = SpikingDifferenceOfExponentials(n_channels=4)
    assert w.analysis.cascade_depths == [1, 1, 1]


def test_spiking_batched_input():
    w = SpikingDifferenceOfExponentials(
        n_channels=3, dt=1e-3, mu_max=0.02, threshold=torch.tensor(0.05)
    )
    spikes = w(torch.randn(128, 4))
    assert spikes.shape == (128, 4, 6)


def test_decode_recovers_the_analysis_coefficients():
    w = SpikingDifferenceOfExponentials(n_channels=4, dt=1e-3, mu_max=0.05)
    t = torch.arange(0, 2.0, 1e-3)
    signal = torch.sin(2 * math.pi * 3 * t)
    truth = w.analysis(signal)
    estimate = w.decode(w(signal)).detach()
    assert estimate.shape == truth.shape
    for k in range(w.n_channels):
        corr = torch.corrcoef(torch.stack([truth[:, k], estimate[:, k]]))[0, 1]
        assert corr > 0.8, f"channel {k} correlation {corr}"


def test_decode_of_silence_is_zero():
    w = SpikingDifferenceOfExponentials(n_channels=3, dt=1e-3, mu_max=0.02)
    assert torch.allclose(
        w.decode(w(torch.zeros(256))), torch.zeros(256, 3), atol=1e-12
    )


def test_reconstruct_goes_through_the_non_spiking_transform():
    w = SpikingDifferenceOfExponentials(n_channels=3, dt=1e-3, mu_max=0.02)
    spikes = w(torch.randn(256))
    assert torch.allclose(
        w.reconstruct(spikes), w.analysis.reconstruct(w.decode(spikes))
    )


def test_oracle_weights_improve_reconstruction():
    w = SpikingDifferenceOfExponentials(n_channels=4, dt=1e-3, mu_max=0.05)
    t = torch.arange(0, 2.0, 1e-3)
    signal = torch.sin(2 * math.pi * 3 * t)
    spikes = w(signal)

    def correlation():
        recon = w.reconstruct(spikes).detach()
        return torch.corrcoef(torch.stack([signal, recon]))[0, 1].item()

    before = correlation()
    assert before > 0.9, f"default decoder correlation {before}"
    w.fit_oracle_weights(signal)
    after = correlation()
    assert after > before
    assert after > 0.97, f"oracle decoder correlation {after}"


def test_oracle_weights_restore_the_amplitude():
    w = SpikingDifferenceOfExponentials(n_channels=4, dt=1e-3, mu_max=0.05)
    t = torch.arange(0, 2.0, 1e-3)
    signal = torch.sin(2 * math.pi * 3 * t)
    w.fit_oracle_weights(signal)
    recon = w.reconstruct(w(signal)).detach()
    assert 0.8 < recon.abs().max() / signal.abs().max() < 1.25


def test_oracle_weights_are_stored_on_the_analysis_transform():
    w = SpikingDifferenceOfExponentials(n_channels=3, dt=1e-3, mu_max=0.02)
    t = torch.arange(0, 1.0, 1e-3)
    weights = w.fit_oracle_weights(torch.sin(2 * math.pi * 4 * t))
    assert torch.allclose(
        w.analysis.channel_weights, weights.to(torch.float32), atol=1e-6
    )


def test_fit_oracle_weights_rejects_batched_signals():
    w = SpikingDifferenceOfExponentials(n_channels=3)
    with pytest.raises(ValueError, match="1-dimensional"):
        w.fit_oracle_weights(torch.randn(2, 128))


def test_encoder_composes_into_a_norse_pipeline_forward_only():
    # The encoder must work as the front of a pipeline without any decoding.
    from norse.torch.module.sequential import SequentialState
    from norse.torch.module.lif import LIFCell

    dt = 1e-3
    t = torch.arange(0, 0.2, dt)
    batch = torch.stack([torch.sin(2 * math.pi * f * t) for f in (3.0, 7.0)], dim=1)
    model = SequentialState(
        SpikingDifferenceOfExponentials(n_channels=4, dt=dt, mu_max=0.05),
        torch.nn.Linear(8, 5),
        LIFCell(),
    )
    out, _ = model(batch)
    assert out.shape == (len(t), 2, 5)


def test_encoder_is_trainable_end_to_end():
    from norse.torch.module.sequential import SequentialState

    dt = 1e-3
    t = torch.arange(0, 0.2, dt)
    model = SequentialState(
        SpikingDifferenceOfExponentials(
            n_channels=3, dt=dt, mu_max=0.02, learn_threshold=True
        ),
        torch.nn.Linear(6, 2),
    )
    out, _ = model(torch.sin(2 * math.pi * 5 * t))
    out.sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_encoder_output_is_time_first():
    dt = 1e-3
    w = SpikingDifferenceOfExponentials(n_channels=3, dt=dt, mu_max=0.02)
    spikes = w(torch.randn(64, 5))
    # (T, batch, 2 * n_channels), matching Norse's temporal convention.
    assert spikes.shape == (64, 5, 6)


def test_analysis_output_is_time_first():
    w = DifferenceOfTimeCausalKernels(n_channels=4, dt=1e-3, mu_max=0.02)
    assert w(torch.randn(64, 5)).shape == (64, 5, 4)


def test_default_decoder_needs_no_reference_signal():
    # reconstruct() must work straight after construction.
    w = SpikingDifferenceOfExponentials(n_channels=4, dt=1e-3, mu_max=0.05)
    assert torch.allclose(w.analysis.channel_weights, torch.ones(4))
    t = torch.arange(0, 1.0, 1e-3)
    signal = torch.sin(2 * math.pi * 3 * t)
    recon = w.reconstruct(w(signal)).detach()
    corr = torch.corrcoef(torch.stack([signal, recon]))[0, 1]
    assert corr > 0.9, f"default decoder correlation {corr}"


def test_oracle_weights_are_signal_dependent():
    # Documented caveat: the fit is specific to the signal, not a fixed gain.
    t = torch.arange(0, 2.0, 1e-3)
    fits = []
    for frequency in (3.0, 7.0):
        w = SpikingDifferenceOfExponentials(n_channels=4, dt=1e-3, mu_max=0.05)
        fits.append(w.fit_oracle_weights(torch.sin(2 * math.pi * frequency * t)))
    assert not torch.allclose(fits[0], fits[1], rtol=0.5)


def test_default_membrane_is_several_samples_long():
    # A membrane of one sample makes the neuron a comparator and saturates it.
    from norse.torch.module.wavelet import MEMBRANE_STEPS

    dt = 1e-3
    w = SpikingDifferenceOfExponentials(n_channels=3, dt=dt, mu_max=0.02)
    assert MEMBRANE_STEPS > 1
    assert torch.allclose(w.mu_mem, torch.full((3,), MEMBRANE_STEPS * dt))


def test_longer_membrane_reduces_spike_density():
    dt = 1e-3
    t = torch.arange(0, 1.0, dt)
    signal = torch.sin(2 * math.pi * 3 * t)
    densities = []
    for steps in (1, 5, 20):
        w = SpikingDifferenceOfExponentials(
            n_channels=4, dt=dt, mu_max=0.05, mu_mem=torch.full((4,), steps * dt)
        )
        densities.append(w(signal).mean().item())
    assert densities[0] > densities[1] > densities[2]


def test_default_membrane_beats_a_single_step_membrane():
    dt = 1e-3
    t = torch.arange(0, 2.0, dt)
    signal = torch.sin(2 * math.pi * 3 * t)

    def correlation(mu_mem):
        w = SpikingDifferenceOfExponentials(
            n_channels=4, dt=dt, mu_max=0.05, mu_mem=mu_mem
        )
        recon = w.reconstruct(w(signal)).detach()
        return torch.corrcoef(torch.stack([signal, recon]))[0, 1].item()

    assert correlation(None) > correlation(torch.full((4,), dt))
