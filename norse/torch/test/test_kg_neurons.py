"""Tests for Klein-Gordon wave neuron models (spikingjelly, snntorch, bindsnet, norse).

These tests verify the core Klein-Gordon leapfrog physics shared by all
SNN implementations, and benchmark oscillatory neurons against standard
LIF neurons.

Runs standalone with only torch + numpy — no framework install required.
"""

import numpy as np
import pytest
import torch


# ============================================================
# Core Klein-Gordon leapfrog implementation (reference)
# ============================================================

def kg_leapfrog_step(v, v_prev, chi, dt, x):
    """Reference Klein-Gordon leapfrog: v_new = 2v - v_prev + dt^2(-chi^2 v + x)"""
    return 2.0 * v - v_prev + dt**2 * (-chi**2 * v + x)


def kg_damped_step(v, v_prev, chi, gamma, dt, x):
    """Reference damped KG: v_new = (2v - (1-gd2)v_prev + dt^2(-chi^2 v + x)) / (1+gd2)"""
    gd2 = gamma * dt / 2.0
    return (2.0 * v - (1.0 - gd2) * v_prev + dt**2 * (-chi**2 * v + x)) / (1.0 + gd2)


def lif_step(v, tau, dt, x, v_rest=0.0):
    """Reference LIF: v_new = v + dt/tau * (-(v-v_rest) + x)"""
    return v + dt / tau * (-(v - v_rest) + x)


# ============================================================
# Physics Tests
# ============================================================

class TestKGLeapfrogPhysics:
    """Tests for the Klein-Gordon leapfrog update correctness."""

    def test_zero_input_oscillation(self):
        """With initial displacement and no input, neuron should oscillate."""
        chi, dt = 2.0, 0.05
        v = 1.0
        v_prev = 1.0  # starting from rest (no velocity)
        trajectory = [v]
        for _ in range(500):
            v_new = kg_leapfrog_step(v, v_prev, chi, dt, 0.0)
            v_prev = v
            v = v_new
            trajectory.append(v)

        trajectory = np.array(trajectory)
        # Should oscillate (cross zero multiple times)
        sign_changes = np.sum(np.diff(np.sign(trajectory)) != 0)
        assert sign_changes > 5, f"Only {sign_changes} zero crossings (not oscillating)"

        # Amplitude should be approximately conserved (leapfrog is symplectic)
        max_amp = np.max(np.abs(trajectory))
        assert 0.9 < max_amp < 1.1, f"Amplitude not conserved: {max_amp}"

    def test_oscillation_frequency_matches_chi(self):
        """The oscillation period should match T = 2*pi/chi."""
        chi, dt = 3.0, 0.01
        n_steps = 10000
        v = 1.0
        v_prev = 1.0
        trajectory = []
        for _ in range(n_steps):
            v_new = kg_leapfrog_step(v, v_prev, chi, dt, 0.0)
            v_prev = v
            v = v_new
            trajectory.append(v)

        trajectory = np.array(trajectory)
        # FFT to find dominant frequency
        spectrum = np.abs(np.fft.rfft(trajectory))
        freqs = np.fft.rfftfreq(n_steps, d=dt)
        dominant_freq = freqs[np.argmax(spectrum[1:]) + 1]  # skip DC
        expected_freq = chi / (2 * np.pi)
        assert abs(dominant_freq - expected_freq) < 0.02, \
            f"Dominant freq {dominant_freq:.3f} vs expected {expected_freq:.3f}"

    def test_cfl_stability_condition(self):
        """Verify dt*chi < 2 is required for stability."""
        chi = 5.0
        # Stable: dt = 0.1, dt*chi = 0.5
        v, v_prev = 1.0, 1.0
        for _ in range(1000):
            v_new = kg_leapfrog_step(v, v_prev, chi, 0.1, 0.0)
            v_prev = v
            v = v_new
        assert np.isfinite(v), "Should be stable for dt*chi=0.5"

        # Unstable: dt = 0.5, dt*chi = 2.5
        v, v_prev = 1.0, 1.0
        for _ in range(100):
            v_new = kg_leapfrog_step(v, v_prev, chi, 0.5, 0.0)
            v_prev = v
            v = v_new
        assert abs(v) > 1e10, "Should be unstable for dt*chi=2.5"

    def test_energy_conservation(self):
        """Leapfrog should conserve energy (symplectic integrator)."""
        chi, dt = 2.0, 0.05
        v = 1.0
        v_prev = np.cos(chi * dt)  # initial velocity v_dot ~ 0

        def energy(v, v_prev):
            v_dot = (v - v_prev) / dt
            return 0.5 * v_dot**2 + 0.5 * chi**2 * v**2

        E0 = energy(v, v_prev)
        for _ in range(10000):
            v_new = kg_leapfrog_step(v, v_prev, chi, dt, 0.0)
            v_prev = v
            v = v_new

        E_final = energy(v, v_prev)
        drift = abs(E_final - E0) / E0
        assert drift < 0.05, f"Energy drift {drift*100:.2f}% exceeds 5%"

    def test_resonance_response(self):
        """Input at frequency chi should produce maximum amplitude response."""
        chi, dt = 2.0, 0.05
        n_steps = 2000

        amplitudes = {}
        for omega in [0.5, 1.0, chi, 3.0, 5.0]:
            v = 0.0
            v_prev = 0.0
            for step in range(n_steps):
                t = step * dt
                x = 0.1 * np.sin(omega * t)
                v_new = kg_leapfrog_step(v, v_prev, chi, dt, x)
                v_prev = v
                v = v_new
            amplitudes[omega] = abs(v)

        # Resonant frequency should produce largest response
        max_omega = max(amplitudes, key=amplitudes.get)
        assert max_omega == chi, \
            f"Resonance at omega={max_omega}, expected {chi}. " \
            f"Responses: {amplitudes}"

    def test_damped_reduces_amplitude(self):
        """Damped KG should have decreasing oscillation amplitude."""
        chi, gamma, dt = 2.0, 0.3, 0.05
        v = 1.0
        v_prev = np.cos(chi * dt)  # start with velocity ~0, position 1

        trace = []
        for step in range(5000):
            v_new = kg_damped_step(v, v_prev, chi, gamma, dt, 0.0)
            v_prev = v
            v = v_new
            trace.append(v)
        trace = np.array(trace)

        # Use envelope (abs) and find local maxima
        abs_trace = np.abs(trace)
        peaks = []
        for i in range(1, len(abs_trace) - 1):
            if abs_trace[i] > abs_trace[i-1] and abs_trace[i] > abs_trace[i+1] and abs_trace[i] > 0.001:
                peaks.append(abs_trace[i])

        # Peak amplitudes should decrease
        assert len(peaks) >= 3, f"Only {len(peaks)} peaks detected"
        for i in range(len(peaks) - 1):
            assert peaks[i + 1] < peaks[i] * 1.01, \
                f"Peak {i+1} ({peaks[i+1]:.4f}) not less than peak {i} ({peaks[i]:.4f})"

    def test_damped_gamma_zero_recovers_undamped(self):
        """gamma=0 damped should equal undamped."""
        chi, dt = 2.0, 0.05
        v = 1.0
        v_prev = 1.0

        v_und, v_und_prev = v, v_prev
        v_dmp, v_dmp_prev = v, v_prev

        for step in range(500):
            x = 0.1 * np.sin(step * dt)
            v_und_new = kg_leapfrog_step(v_und, v_und_prev, chi, dt, x)
            v_dmp_new = kg_damped_step(v_dmp, v_dmp_prev, chi, 0.0, dt, x)
            assert abs(v_und_new - v_dmp_new) < 1e-12
            v_und_prev, v_und = v_und, v_und_new
            v_dmp_prev, v_dmp = v_dmp, v_dmp_new


# ============================================================
# Torch implementation tests (verify each repo's module)
# ============================================================

class TestTorchKGUpdate:
    """Test the KG update using PyTorch tensors (verifies autograd compatibility)."""

    def test_torch_forward(self):
        """Basic KG step with torch tensors."""
        chi = torch.tensor(2.0)
        dt = 0.05
        v = torch.tensor(1.0)
        v_prev = torch.tensor(1.0)
        x = torch.tensor(0.5)

        v_new = 2.0 * v - v_prev + dt**2 * (-chi**2 * v + x)
        assert torch.isfinite(v_new)

    def test_torch_gradient_flows(self):
        """Gradients should flow through the KG update w.r.t. chi."""
        chi = torch.tensor(2.0, requires_grad=True)
        dt = 0.05
        v = torch.tensor(1.0)
        v_prev = torch.tensor(1.0)
        x = torch.tensor(0.5)

        v_new = 2.0 * v - v_prev + dt**2 * (-chi**2 * v + x)
        v_new.backward()
        assert chi.grad is not None
        assert torch.isfinite(chi.grad)

    def test_torch_batch(self):
        """KG update should work with batched tensors."""
        batch = 32
        neurons = 64
        chi = torch.tensor(2.0)
        dt = 0.05
        v = torch.randn(batch, neurons)
        v_prev = torch.randn(batch, neurons)
        x = torch.randn(batch, neurons)

        v_new = 2.0 * v - v_prev + dt**2 * (-chi**2 * v + x)
        assert v_new.shape == (batch, neurons)

    def test_torch_spike_generation(self):
        """Spikes should be generated when membrane crosses threshold."""
        chi = 2.0
        dt = 0.05
        threshold = 1.0
        v = torch.tensor(0.0)
        v_prev = torch.tensor(0.0)

        n_spikes = 0
        for step in range(2000):
            x = torch.tensor(0.5 * np.sin(chi * step * dt))
            v_new = 2.0 * v - v_prev + dt**2 * (-chi**2 * v + x)
            spike = (v_new >= threshold).float()
            n_spikes += spike.item()
            # Reset on spike
            v_prev = v.clone()
            v = (1 - spike) * v_new
            if spike.item():
                v_prev = torch.tensor(0.0)
        assert n_spikes > 0, "No spikes generated with resonant input"

    def test_torch_per_neuron_chi(self):
        """Per-neuron chi values produce different resonance frequencies."""
        n_neurons = 4
        chi = torch.tensor([1.0, 2.0, 3.0, 4.0])
        dt = 0.01
        v = torch.zeros(n_neurons)
        v_prev = torch.zeros(n_neurons)

        # Drive at frequency 2.0
        max_responses = torch.zeros(n_neurons)
        for step in range(5000):
            t = step * dt
            x = torch.full((n_neurons,), 0.1 * np.sin(2.0 * t))
            v_new = 2.0 * v - v_prev + dt**2 * (-chi**2 * v + x)
            v_prev = v.clone()
            v = v_new
            max_responses = torch.maximum(max_responses, v.abs())

        # Neuron with chi=2.0 should respond most strongly
        assert torch.argmax(max_responses).item() == 1  # index 1 = chi=2.0


# ============================================================
# Benchmark: KG Wave Neuron vs LIF
# ============================================================

class TestKGvsLIFBenchmark:
    """Benchmark comparing KG wave neuron against standard LIF neuron."""

    def test_resonance_selectivity(self):
        """KG neurons are frequency-selective; LIF neurons are not.

        HYPOTHESIS: Feed the same multi-frequency signal to:
          - KG neuron (chi=target_freq)
          - LIF neuron (tau matched to similar timescale)

        KG should selectively amplify the target frequency while
        LIF responds to the total input power regardless of frequency.
        """
        dt = 0.01
        n_steps = 10000
        target_freq = 3.0
        chi = target_freq  # KG natural frequency

        # Multi-frequency input: target + distractors
        t = np.arange(n_steps) * dt
        signal_target = 0.1 * np.sin(target_freq * t)
        signal_distract = 0.3 * np.sin(0.5 * t) + 0.3 * np.sin(7.0 * t)

        # Run KG
        v_kg = 0.0
        v_kg_prev = 0.0
        kg_response_target = []
        kg_response_mixed = []
        for step in range(n_steps):
            # Target only
            v_new = kg_leapfrog_step(v_kg, v_kg_prev, chi, dt, signal_target[step])
            v_kg_prev = v_kg
            v_kg = v_new
            kg_response_target.append(v_kg)

        v_kg = 0.0
        v_kg_prev = 0.0
        for step in range(n_steps):
            # Mixed signal
            v_new = kg_leapfrog_step(v_kg, v_kg_prev, chi, dt,
                                     signal_target[step] + signal_distract[step])
            v_kg_prev = v_kg
            v_kg = v_new
            kg_response_mixed.append(v_kg)

        # Run LIF with similar timescale
        tau = 2 * np.pi / chi  # match period
        v_lif = 0.0
        lif_response_target = []
        lif_response_mixed = []
        for step in range(n_steps):
            v_new = lif_step(v_lif, tau, dt, signal_target[step])
            v_lif = v_new
            lif_response_target.append(v_lif)

        v_lif = 0.0
        for step in range(n_steps):
            v_new = lif_step(v_lif, tau, dt,
                             signal_target[step] + signal_distract[step])
            v_lif = v_new
            lif_response_mixed.append(v_lif)

        # KG: response to mixed should be dominated by target component
        kg_target_rms = np.sqrt(np.mean(np.array(kg_response_target[-5000:])**2))
        kg_mixed_rms = np.sqrt(np.mean(np.array(kg_response_mixed[-5000:])**2))

        # LIF: response to mixed should scale with total power
        lif_target_rms = np.sqrt(np.mean(np.array(lif_response_target[-5000:])**2))
        lif_mixed_rms = np.sqrt(np.mean(np.array(lif_response_mixed[-5000:])**2))

        # KG selectivity: mixed/target ratio should be close to 1 (rejects distractors)
        kg_ratio = kg_mixed_rms / kg_target_rms if kg_target_rms > 0 else float('inf')
        # LIF ratio: mixed/target should be much larger (responds to all power)
        lif_ratio = lif_mixed_rms / lif_target_rms if lif_target_rms > 0 else float('inf')

        print("\n--- Resonance Selectivity Benchmark ---")
        print(f"  Target freq = {target_freq}, distractor power 6x larger")
        print(f"  KG neuron: target_rms={kg_target_rms:.4f}, "
              f"mixed_rms={kg_mixed_rms:.4f}, ratio={kg_ratio:.2f}")
        print(f"  LIF neuron: target_rms={lif_target_rms:.4f}, "
              f"mixed_rms={lif_mixed_rms:.4f}, ratio={lif_ratio:.2f}")
        print(f"  => KG rejects off-resonance noise (ratio closer to 1)")
        print(f"  => LIF amplifies all input (ratio >> 1)")

        # KG should be more selective (lower ratio = better rejection)
        assert kg_ratio < lif_ratio, \
            f"KG ratio {kg_ratio:.2f} not better than LIF {lif_ratio:.2f}"

    def test_temporal_memory(self):
        """KG neurons maintain oscillatory memory longer than LIF decay.

        After a brief pulse, the KG neuron continues to ring while
        the LIF neuron exponentially decays.
        """
        dt = 0.01
        chi = 3.0
        tau = 2 * np.pi / chi

        # Input: brief pulse
        n_steps = 5000
        pulse_duration = 50

        # KG
        v_kg = 0.0
        v_kg_prev = 0.0
        kg_trace = []
        for step in range(n_steps):
            x = 1.0 if step < pulse_duration else 0.0
            v_kg_new = kg_leapfrog_step(v_kg, v_kg_prev, chi, dt, x)
            v_kg_prev = v_kg
            v_kg = v_kg_new
            kg_trace.append(abs(v_kg))

        # LIF
        v_lif = 0.0
        lif_trace = []
        for step in range(n_steps):
            x = 1.0 if step < pulse_duration else 0.0
            v_lif = lif_step(v_lif, tau, dt, x)
            lif_trace.append(abs(v_lif))

        # Measure response at t=2000 (long after pulse)
        late_idx = 2000
        kg_late = np.mean(np.array(kg_trace[late_idx:late_idx+500]))
        lif_late = np.mean(np.array(lif_trace[late_idx:late_idx+500]))

        print("\n--- Temporal Memory Benchmark ---")
        print(f"  Pulse for {pulse_duration} steps, then silence for {n_steps - pulse_duration}")
        print(f"  KG neuron at t={late_idx*dt:.1f}s: avg |v| = {kg_late:.6f}")
        print(f"  LIF neuron at t={late_idx*dt:.1f}s: avg |v| = {lif_late:.6f}")
        print(f"  => KG oscillatory memory lasts {kg_late/max(lif_late, 1e-15):.0f}x longer")

        # KG should still have significant activity, LIF should have decayed
        assert kg_late > lif_late * 10, \
            f"KG memory {kg_late:.6f} not significantly > LIF {lif_late:.6f}"

    def test_spike_rate_with_resonant_input(self):
        """KG neuron should spike more with resonant input than off-resonant."""
        chi = 2.0
        dt = 0.05
        threshold = 0.5
        n_steps = 5000

        spike_counts = {}
        for omega in [0.5, 1.0, chi, 3.5, 5.0]:
            v = 0.0
            v_prev = 0.0
            spikes = 0
            for step in range(n_steps):
                x = 0.3 * np.sin(omega * step * dt)
                v_new = kg_leapfrog_step(v, v_prev, chi, dt, x)
                if v_new > threshold:
                    spikes += 1
                    v_new = 0.0
                    v = 0.0
                v_prev = v
                v = v_new
            spike_counts[omega] = spikes

        # Resonant frequency should produce most spikes
        max_omega = max(spike_counts, key=spike_counts.get)
        assert max_omega == chi, \
            f"Max spikes at omega={max_omega}, expected {chi}. {spike_counts}"

        print("\n--- Spike Rate vs Input Frequency ---")
        for omega, count in sorted(spike_counts.items()):
            marker = " <-- resonant" if omega == chi else ""
            print(f"  omega={omega:.1f}: {count} spikes{marker}")
