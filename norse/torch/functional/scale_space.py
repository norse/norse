r"""Primitives for Lindeberg's time-causal temporal scale-space.

Two conventions are used throughout and are easy to confuse:

* **variance** :math:`\tau` -- the second moment of the smoothing kernel,
  which is what composes additively down a cascade.
* **scale** :math:`\mu = \sqrt\tau` -- the time constant of a single
  truncated exponential, in seconds.

.. [Lindeberg2016] Lindeberg 2016, Time-Causal and Time-Recursive
    Spatio-Temporal Receptive Fields,
    https://link.springer.com/article/10.1007/s10851-015-0613-9
"""

import math

import torch


def geometric_variance_ladder(
    n_levels: int, c: float, max_variance: float
) -> torch.Tensor:
    r"""Geometric ladder of scale variances, coarsest level last.

    The distribution underlying both the temporal receptive fields and the
    wavelet scale levels ([Lindeberg2016], Eq. 58):

    .. math:: \tau_k = c^{2(k - K)}\,\tau_{\max}, \qquad k = 1 \dots K.

    Args:
        n_levels: Number of levels :math:`K`.
        c: Ratio between adjacent scales. ``c > 1`` gives an ascending
            ladder ending at ``max_variance``.
        max_variance: The coarsest variance :math:`\tau_{\max}`.

    Returns:
        Tensor of shape ``(n_levels,)``, ascending for ``c > 1``, whose last
        entry is ``max_variance``.
    """
    k = torch.linspace(1, n_levels, n_levels)
    return c ** (2 * (k - n_levels)) * max_variance


def exponential_tau_mem_inv(mu: torch.Tensor, dt: float) -> torch.Tensor:
    r"""``tau_mem_inv`` giving *exact* exponential decay per step.

    Norse's leaky integrators advance by forward Euler,

    .. math:: v[n+1] = (1 - \Delta t\,\tau^{-1})\,v[n]
                       + \Delta t\,\tau^{-1}\,i[n],

    and the truncated exponential underlying the time-causal scale-space
    decays by exactly :math:`\alpha = e^{-\Delta t/\mu}` per step. Equating
    the two multipliers,

    .. math:: \Delta t\,\tau^{-1} = 1 - e^{-\Delta t/\mu},

    gives the exponential filter exactly. ``expm1`` keeps the right-hand
    side accurate when :math:`\Delta t \ll \mu`.

    Note:
        The result is the per-step leak fraction divided by ``dt``, so it
        is unbounded above and must be paired with the ``dt`` it was built
        with. :func:`per_step_leak` returns the bounded quantity.

    Args:
        mu: Scale(s) :math:`\mu` in seconds, strictly positive.
        dt: Integration time step in seconds.

    Returns:
        ``tau_mem_inv`` for :class:`norse.torch.LIBoxParameters` and
        friends, same shape as ``mu``.
    """
    return per_step_leak(mu, dt) / dt


def per_step_leak(mu: torch.Tensor, dt: float) -> torch.Tensor:
    r"""The per-step leak fraction :math:`1 - e^{-\Delta t/\mu}`.

    This is the quantity a leaky integrator actually multiplies by, and it
    lies in :math:`(0, 1)` for every :math:`\mu > 0`, so a divergent filter
    cannot be expressed. It is also well conditioned for slow neurons:
    recovering :math:`\mu` from a float32 leak fraction is accurate to
    ~3e-8 relative at :math:`\mu = 100\,\mathrm{s}`.

    Invert with :func:`mu_from_per_step_leak`.

    Args:
        mu: Scale(s) :math:`\mu` in seconds, strictly positive.
        dt: Integration time step in seconds.

    Returns:
        Leak fraction in ``(0, 1)``, same shape as ``mu``.
    """
    mu = torch.as_tensor(mu)
    if torch.any(mu <= 0):
        raise ValueError(f"mu must be strictly positive, got {mu}")
    return -torch.expm1(-dt / mu)


def mu_from_per_step_leak(leak: torch.Tensor, dt: float) -> torch.Tensor:
    r"""Recover the scale :math:`\mu` from a per-step leak fraction.

    Inverse of :func:`per_step_leak`:
    :math:`\mu = -\Delta t / \log(1 - \text{leak})`.

    Args:
        leak: Per-step leak fraction in ``(0, 1)``.
        dt: Integration time step in seconds.

    Returns:
        Scale :math:`\mu` in seconds, same shape as ``leak``.
    """
    leak = torch.as_tensor(leak)
    if torch.any(leak <= 0) or torch.any(leak >= 1):
        raise ValueError(f"leak must lie strictly in (0, 1), got {leak}")
    return -dt / torch.log1p(-leak)


def geometric_time_constants(
    mu: float, dt: float, c: float, n_stages: int
) -> torch.Tensor:
    r"""Time constants of one cascade approximating the limit kernel.

    Where :func:`geometric_variance_ladder` spreads scales *across*
    channels, this splits a *single* scale :math:`\mu` into ``n_stages``
    cascaded truncated exponentials whose composition approximates
    Lindeberg's time-causal limit kernel :math:`\Psi(t;\mu,c)`.

    Working in sample units, stage :math:`i` of :math:`N` carries
    cumulative variance :math:`\tau_i = \sigma^2 / c^{2(N-i-1)}`; the
    stage's own scale follows by inverting :math:`\tau = \mu^2 + \mu` on the
    increment :math:`\Delta\tau`, and is converted back to seconds.

    Args:
        mu: Target scale :math:`\mu` in seconds.
        dt: Sample period in seconds.
        c: Intra-cascade distribution parameter, ``c > 1``.
        n_stages: Number of cascade stages :math:`N`.

    Returns:
        Tensor of ``n_stages`` time constants in seconds, finest first.

    Raises:
        ValueError: If ``c <= 1``, ``n_stages < 1``, or ``mu == 0``, which
            leaves a stage with no variance to carry.
    """
    if c <= 1:
        raise ValueError(f"c must be > 1, got {c}")
    if n_stages < 1:
        raise ValueError(f"n_stages must be >= 1, got {n_stages}")

    sigma_samples = float(mu) / float(dt)
    out = torch.empty(n_stages, dtype=torch.float64)
    tau_prev = 0.0
    for i in range(n_stages):
        tau_cur = sigma_samples**2 / c ** (2 * (n_stages - i - 1))
        dtau = tau_cur - tau_prev
        mu_stage = (-1.0 + math.sqrt(1.0 + 4.0 * dtau)) / 2.0
        if mu_stage <= 0.0:
            raise ValueError(
                f"Non-positive stage scale {mu_stage} at stage {i} "
                f"(sigma_samples={sigma_samples}, n_stages={n_stages}, c={c}). "
                "Increase mu or reduce n_stages."
            )
        out[i] = dt / math.log1p(1.0 / mu_stage)
        tau_prev = tau_cur
    return out


def stable_cascade_depth(
    mu: float, dt: float, max_depth: int, alpha_floor: float = 0.01
) -> int:
    r"""Largest cascade depth :math:`N \le` ``max_depth`` that stays stable.

    The bound is :math:`N \le (\mu / \mu_{\min})^2` with
    :math:`\mu_{\min} = \Delta t / (-\log \alpha_{\text{floor}})`, which is
    the depth at which a *uniform* cascade -- every stage at
    :math:`\mu/\sqrt N` -- still retains ``alpha_floor`` per stage.

    Warning:
        The bound does not carry over to the geometric cascade of
        :func:`geometric_time_constants`, whose finest stages are much
        shorter than :math:`\mu/\sqrt N`. At :math:`\mu = 2\,\mathrm{ms}`,
        :math:`\Delta t = 1\,\mathrm{ms}` and ``max_depth=12`` this returns
        12, for which the uniform stage retains 0.18 per step but the
        finest geometric stage retains 1e-6 and so passes its input
        straight through. Choose ``max_depth`` against the retentions
        :func:`geometric_time_constants` actually produces.

    Args:
        mu: Target scale in seconds.
        dt: Sample period in seconds.
        max_depth: Upper bound on the depth.
        alpha_floor: Smallest per-stage retention of the uniform cascade.

    Returns:
        Depth in ``[1, max_depth]``.
    """
    mu_stage_min = dt / (-math.log(alpha_floor))
    depth_stable = max(1, int((float(mu) / mu_stage_min) ** 2))
    return max(1, min(max_depth, depth_stable))
