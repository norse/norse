import math

import numpy as np
import pytest
import torch

from norse.torch.functional.scale_space import (
    exponential_tau_mem_inv,
    geometric_time_constants,
    geometric_variance_ladder,
    mu_from_per_step_leak,
    per_step_leak,
    stable_cascade_depth,
)


def test_variance_ladder_ends_at_max():
    ladder = geometric_variance_ladder(5, 2.0, 16.0)
    assert ladder.shape == (5,)
    assert ladder[-1].item() == pytest.approx(16.0)
    # ratio between adjacent levels is c^2
    ratios = (ladder[1:] / ladder[:-1]).tolist()
    assert all(r == pytest.approx(4.0) for r in ratios)


def test_variance_ladder_ascends_for_c_above_one():
    ladder = geometric_variance_ladder(6, 1.5, 10.0)
    assert torch.all(ladder[1:] > ladder[:-1])


def test_variance_ladder_single_level():
    assert geometric_variance_ladder(1, 2.0, 3.0).tolist() == pytest.approx([3.0])


@pytest.mark.parametrize("mu", [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 100.0])
def test_per_step_leak_is_bounded(mu):
    leak = per_step_leak(torch.as_tensor(mu), dt=1e-3).item()
    assert 0.0 < leak < 1.0


def test_per_step_leak_matches_closed_form():
    dt = 1e-3
    mu = torch.tensor([1e-3, 5e-3, 2e-2], dtype=torch.float64)
    expected = 1.0 - torch.exp(-dt / mu)
    assert torch.allclose(per_step_leak(mu, dt), expected)


@pytest.mark.parametrize("mu", [1e-3, 1e-2, 1.0, 10.0, 100.0])
def test_leak_round_trip(mu):
    dt = 1e-3
    leak = per_step_leak(torch.as_tensor(mu, dtype=torch.float64), dt)
    recovered = mu_from_per_step_leak(leak, dt)
    assert recovered.item() == pytest.approx(mu, rel=1e-12)


def test_leak_round_trip_float32_precision_at_long_tau():
    # The leak fraction stays small for slow neurons, where alpha itself
    # would crowd against 1.0 and lose resolution in float32.
    dt, mu = 1e-3, 100.0
    leak32 = per_step_leak(torch.tensor(mu), dt).to(torch.float32)
    recovered = mu_from_per_step_leak(leak32.double(), dt).item()
    assert abs(recovered / mu - 1.0) < 1e-6


def test_exponential_tau_mem_inv_reproduces_exact_decay():
    dt, mu, T = 1e-3, 5e-3, 200
    alpha = math.exp(-dt / mu)
    x = torch.randn(T, dtype=torch.float64)

    reference = torch.empty(T, dtype=torch.float64)
    v = torch.zeros((), dtype=torch.float64)
    for n in range(T):
        v = alpha * v + (1 - alpha) * x[n]
        reference[n] = v

    # The Euler update Norse applies, with the converted parameter.
    tau_inv = exponential_tau_mem_inv(torch.as_tensor(mu, dtype=torch.float64), dt)
    got = torch.empty(T, dtype=torch.float64)
    v = torch.zeros((), dtype=torch.float64)
    for n in range(T):
        v = v + dt * tau_inv * ((0.0 - v) + x[n])
        got[n] = v

    assert torch.allclose(got, reference, atol=1e-12)


def test_exponential_tau_mem_inv_stays_stable_where_naive_diverges():
    # dt/mu = 4.6 is reachable by the finest cascade stages.
    dt = 1e-3
    mu = dt / (-math.log(0.01))
    assert dt / mu > 2  # the naive 1/mu would give |1 - dt/mu| > 1
    multiplier = 1 - dt * exponential_tau_mem_inv(torch.as_tensor(mu), dt).item()
    assert 0.0 < multiplier < 1.0
    assert multiplier == pytest.approx(0.01, abs=1e-6)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_per_step_leak_rejects_non_positive_mu(bad):
    with pytest.raises(ValueError, match="strictly positive"):
        per_step_leak(torch.as_tensor(bad), dt=1e-3)


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.1])
def test_mu_from_leak_rejects_out_of_range(bad):
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        mu_from_per_step_leak(torch.as_tensor(bad), dt=1e-3)


def test_geometric_time_constants_matches_lindeberg_recurrence():
    dt, c, mu, n = 1e-3, 2.0, 0.05, 4

    def reference(mu, dt, c, n):
        sigma = mu / dt
        out, prev = np.empty(n), 0.0
        for i in range(n):
            cur = sigma * sigma / c ** (2 * (n - i - 1))
            stage = (-1.0 + math.sqrt(1.0 + 4.0 * (cur - prev))) / 2.0
            out[i] = dt / math.log1p(1.0 / stage)
            prev = cur
        return out

    got = geometric_time_constants(mu, dt, c, n).numpy()
    assert np.allclose(got, reference(mu, dt, c, n), rtol=0, atol=0)


def test_geometric_time_constants_ordered_finest_first():
    taus = geometric_time_constants(0.2, 1e-3, 2.0, 7)
    assert torch.all(taus[1:] > taus[:-1])


def test_geometric_time_constants_rejects_bad_arguments():
    with pytest.raises(ValueError, match="c must be > 1"):
        geometric_time_constants(0.1, 1e-3, 1.0, 3)
    with pytest.raises(ValueError, match="n_stages must be >= 1"):
        geometric_time_constants(0.1, 1e-3, 2.0, 0)
    with pytest.raises(ValueError, match="Non-positive stage scale"):
        geometric_time_constants(0.0, 1e-3, 2.0, 5)


def test_geometric_time_constants_accepts_arbitrarily_fine_scales():
    # dtau is strictly positive for c > 1, so every stage scale is positive
    # however small mu becomes.
    taus = geometric_time_constants(1e-9, 1e-3, 2.0, 5)
    assert torch.all(taus > 0)


def test_stable_cascade_depth_is_clamped():
    dt = 1e-3
    assert stable_cascade_depth(1e-4, dt, max_depth=7) == 1  # too fine to cascade
    assert stable_cascade_depth(10.0, dt, max_depth=7) == 7  # capped by max_depth
    assert 1 <= stable_cascade_depth(5e-3, dt, max_depth=7) <= 7


def test_stable_cascade_depth_bounds_the_uniform_cascade():
    dt, max_depth, alpha_floor = 1e-3, 12, 0.01
    for mu in [2e-3, 5e-3, 2e-2, 1e-1]:
        n = stable_cascade_depth(mu, dt, max_depth, alpha_floor)
        uniform_stage = mu / math.sqrt(n)
        assert math.exp(-dt / uniform_stage) >= alpha_floor


def test_stable_cascade_depth_does_not_bound_the_geometric_cascade():
    # Documented limitation: the bound is derived for a uniform cascade, so
    # the finest geometric stage can fall far below alpha_floor.
    dt, mu = 1e-3, 2e-3
    n = stable_cascade_depth(mu, dt, max_depth=12, alpha_floor=0.01)
    alphas = torch.exp(-dt / geometric_time_constants(mu, dt, 2.0, n))
    assert alphas.min().item() < 0.01
