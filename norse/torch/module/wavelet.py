r"""Time-causal, scale-covariant wavelet transforms.

A bank of cascaded truncated exponentials at a geometric ladder of temporal
scales, differenced between adjacent scales to give bandpass channels plus a
lowpass residual. Reconstruction telescopes the channels, so analysis is
exactly invertible.

The transform is evaluated by convolving against the cascade's impulse
response, which costs :math:`O(T \log T)` for the whole signal. The same
filters can be run as a recursion with
:class:`norse.torch.LIBoxCell`, which is what the spiking variants and NIR
export need; see
:func:`norse.torch.functional.scale_space.exponential_tau_mem_inv`.

Implements the spiking bandpass wavelets of Pedersen, Lindeberg and
Gerstoft, *Encoding and Decoding Temporal Signals with Spiking Bandpass
Wavelets*, arXiv:2605.09770 (2026), ported from the Jax reference
implementation at https://github.com/jegp/swavelet.
"""

from typing import Optional

import torch

from norse.torch.functional.scale_space import (
    exponential_tau_mem_inv,
    geometric_variance_ladder,
    geometric_time_constants,
    stable_cascade_depth,
)
from norse.torch.functional.wavelet import (
    bandpass_kernel_norms,
    dot_analysis,
    dot_reconstruct,
    fft_convolve_causal,
    invert_membrane_integration,
    time_causal_limit_kernel,
)
from norse.torch.module.leaky_integrator_box import LIBoxCell, LIBoxParameters
from norse.torch.module.lif_box import LIFBoxCell, LIFBoxParameters

# Default membrane length in samples. See the mu_mem documentation on
# SpikingDifferenceOfTimeCausalKernels for how this value was chosen.
MEMBRANE_STEPS = 5


def _channels_to_time_first(tensor: torch.Tensor) -> torch.Tensor:
    """``(C, *batch, T)`` -> ``(T, *batch, C)``."""
    return tensor.movedim(0, -1).movedim(-2, 0)


def _time_first_to_channels(tensor: torch.Tensor) -> torch.Tensor:
    """``(T, *batch, C)`` -> ``(C, *batch, T)``."""
    return tensor.movedim(-1, 0).movedim(1, -1)


class DifferenceOfTimeCausalKernels(torch.nn.Module):
    r"""Difference-of-time-causal-kernels (DoT) wavelet transform.

    Builds ``n_channels - 1`` smoothing scales
    :math:`\mu_1 < \dots < \mu_K` ending at ``mu_max``, each approximated by
    a cascade of truncated exponentials, and returns the adjacent
    differences together with the coarsest lowpass response:

    .. math::
        \Delta L_1 = L_1 - f, \qquad
        \Delta L_k = L_k - L_{k-1}, \qquad
        x_K = L_K,

    taking the input as the implicit finest level :math:`L_0 = f`. The
    filters are causal, so every output sample depends only on present and
    past input.

    Parameters:
        n_channels (int): Number of output channels: ``n_channels - 1``
            bandpass plus one lowpass residual. Must be at least 2.
        dt (float): Sample period in seconds.
        mu_max (float): Coarsest smoothing scale in seconds.
        c (float): Ratio between adjacent scales, ``c > 1``.
        cascade_depth_max (int): Upper bound on cascade stages per channel.
            The depth actually used per channel is clamped by
            :func:`~norse.torch.functional.scale_space.stable_cascade_depth`.
        alpha_floor (float): Smallest per-stage retention used when choosing
            the cascade depth.
        kernel_length (Optional[int]): Impulse-response length in samples.
            Defaults to the signal length at each call, which captures the
            full response. Set it to bound memory on long signals, at the
            cost of truncating the coarsest scale.
        learn_channel_weights (bool): Make the reconstruction weights a
            learnable parameter. Defaults to False, where they are fixed at
            one and reconstruction is the exact telescoping sum.
    """

    def __init__(
        self,
        n_channels: int = 5,
        dt: float = 1e-3,
        mu_max: float = 0.2,
        c: float = 2.0,
        cascade_depth_max: int = 7,
        alpha_floor: float = 0.01,
        kernel_length: Optional[int] = None,
        learn_channel_weights: bool = False,
    ):
        super().__init__()
        if n_channels < 2:
            raise ValueError(
                "n_channels must be >= 2, giving at least one bandpass "
                f"channel and one lowpass residual, got {n_channels}"
            )
        if c <= 1:
            raise ValueError(f"c must be > 1, got {c}")

        self.n_channels = n_channels
        self.dt = dt
        self.mu_max = mu_max
        self.c = c
        self.cascade_depth_max = cascade_depth_max
        self.alpha_floor = alpha_floor
        self.kernel_length = kernel_length

        n_smooth = n_channels - 1
        mus = geometric_variance_ladder(n_smooth, c, mu_max**2).sqrt()
        self.register_buffer("mus", mus)

        self.cascade_depths = [
            stable_cascade_depth(float(mu), dt, cascade_depth_max, alpha_floor)
            for mu in mus
        ]
        self.cascade_taus = [
            geometric_time_constants(float(mu), dt, c, depth)
            for mu, depth in zip(mus, self.cascade_depths)
        ]
        # Effective scale of each cascade, as the synthesis normalisation needs.
        self.register_buffer(
            "mu_eff",
            torch.stack([t.pow(2).sum().sqrt() for t in self.cascade_taus]).float(),
        )

        weights = torch.ones(n_channels)
        if learn_channel_weights:
            self.channel_weights = torch.nn.Parameter(weights)
        else:
            self.register_buffer("channel_weights", weights)

        self._kernel_cache: dict = {}

    def kernels(self, length: int, device=None, dtype=None) -> torch.Tensor:
        r"""Impulse responses of the K smoothing cascades, shape ``(K, length)``.

        Cached per ``(length, device, dtype)``, since the kernels depend only
        on the scales and the sample period.
        """
        dtype = dtype or torch.get_default_dtype()
        key = (length, str(device), dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = torch.stack(
                [
                    time_causal_limit_kernel(
                        taus.tolist(), self.dt, length, dtype=dtype, device=device
                    )
                    for taus in self.cascade_taus
                ]
            )
        return self._kernel_cache[key]

    def time_scales(self) -> torch.Tensor:
        """The smoothing scales :math:`\\mu_1 \\dots \\mu_K` in seconds."""
        return self.mus

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Analyse a signal, mapping ``(T, *batch)`` to ``(T, *batch, n_channels)``.

        Time comes first and channels last, as elsewhere in Norse, so the
        result feeds straight into other temporal modules. The functional
        layer underneath works on a trailing time axis, which is where the
        FFT wants it.
        """
        signal = input_tensor.movedim(0, -1)
        length = self.kernel_length or signal.shape[-1]
        kernels = self.kernels(length, device=signal.device, dtype=signal.dtype)
        return _channels_to_time_first(dot_analysis(signal, kernels))

    def reconstruct(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`forward`, ``(T, *batch, n_channels)`` -> ``(T, *batch)``."""
        channels_first = _time_first_to_channels(coefficients)
        return dot_reconstruct(channels_first, self.channel_weights).movedim(-1, 0)

    def integrator_parameters(self) -> LIBoxParameters:
        r"""Parameters running the *finest* stage of every cascade as a recursion.

        Gives one :class:`~norse.torch.LIBoxParameters` whose
        ``tau_mem_inv`` has one entry per channel, decaying by exactly
        :math:`e^{-\Delta t/\tau}` per step. Cascades deeper than one stage
        need one such integrator per stage; see :meth:`integrators`.
        """
        taus = torch.stack([t[0] for t in self.cascade_taus]).float()
        return LIBoxParameters(
            tau_mem_inv=exponential_tau_mem_inv(taus, self.dt),
            v_leak=torch.zeros_like(taus),
        )

    def integrators(self) -> torch.nn.ModuleList:
        r"""One :class:`~norse.torch.LIBoxCell` per cascade stage, per channel.

        Returns a :class:`~torch.nn.ModuleList` of ``K`` ``ModuleList``s, the
        ``k``-th holding the cells for channel ``k``'s cascade in order. Each
        cell decays by exactly :math:`e^{-\Delta t/\tau}` per step, so
        stepping a channel's cells in sequence reproduces that channel's
        kernel convolution sample by sample.
        """
        return torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [
                        LIBoxCell(
                            LIBoxParameters(
                                tau_mem_inv=exponential_tau_mem_inv(
                                    tau.float(), self.dt
                                ),
                                v_leak=torch.as_tensor(0.0),
                            ),
                            dt=self.dt,
                        )
                        for tau in taus
                    ]
                )
                for taus in self.cascade_taus
            ]
        )

    def to_nir(self):
        """Export as a :class:`nir.NIRGraph` of ``n_channels`` outputs."""
        from norse.torch.utils.export_nir_wavelet import from_wavelet

        return from_wavelet(self)

    def extra_repr(self) -> str:
        return (
            f"n_channels={self.n_channels}, dt={self.dt}, mu_max={self.mu_max}, "
            f"c={self.c}, cascade_depths={self.cascade_depths}"
        )


class DifferenceOfExponentials(DifferenceOfTimeCausalKernels):
    r"""Difference-of-exponentials (DoE) wavelet transform.

    :class:`DifferenceOfTimeCausalKernels` with a single exponential per
    scale instead of a cascade, which is the cheapest causal variant and the
    one that maps most directly onto a bank of leaky integrators.

    Parameters:
        n_channels (int): Number of output channels, at least 2.
        dt (float): Sample period in seconds.
        mu_max (float): Coarsest smoothing scale in seconds.
        c (float): Ratio between adjacent scales, ``c > 1``.
        kernel_length (Optional[int]): Impulse-response length in samples.
        learn_channel_weights (bool): Make reconstruction weights learnable.
    """

    def __init__(
        self,
        n_channels: int = 3,
        dt: float = 1e-3,
        mu_max: float = 0.2,
        c: float = 2.0**0.5,
        kernel_length: Optional[int] = None,
        learn_channel_weights: bool = False,
    ):
        super().__init__(
            n_channels=n_channels,
            dt=dt,
            mu_max=mu_max,
            c=c,
            cascade_depth_max=1,
            kernel_length=kernel_length,
            learn_channel_weights=learn_channel_weights,
        )


class SpikingDifferenceOfTimeCausalKernels(torch.nn.Module):
    r"""Spiking difference-of-time-causal-kernels (DoT) wavelet encoder.

    Wraps :class:`DifferenceOfTimeCausalKernels` and encodes each analysis
    channel into spikes with a pair of
    :class:`~norse.torch.LIFBoxCell` neurons. A threshold fires only on
    positive crossings while every analysis channel is signed, so channel
    :math:`k` drives one neuron with :math:`+x_k` and one with
    :math:`-x_k`; the output stacks the pairs as
    ``[pos_0, neg_0, pos_1, neg_1, ...]``, making it
    ``2 * n_channels`` wide.

    Channels are divided by their kernels' L2 norms before thresholding, so
    one threshold scale means the same thing at every temporal scale.

    Parameters:
        n_channels (int): Analysis channels, at least 2. The spiking output
            has twice as many rows.
        dt (float): Sample period in seconds.
        mu_max (float): Coarsest smoothing scale in seconds.
        c (float): Ratio between adjacent scales, ``c > 1``.
        cascade_depth_max (int): Upper bound on cascade stages per channel.
        threshold (Optional[torch.Tensor]): Firing threshold per channel.
            Defaults to ``10 * mu * sqrt(cascade_depth)`` per channel.
        mu_mem (Optional[torch.Tensor]): Membrane time constant per channel.
            Defaults to ``5 * dt``. At ``dt`` the membrane settles within a
            sample, so each neuron compares the prescaled coefficient against
            the threshold rather than integrating it, and a channel driven
            well above threshold saturates at one spike per step. Integrating
            over a few samples instead costs delay but keeps the spike train
            informative: on a 3 Hz sine with four channels the default
            decoder reaches 0.95 at ``5 * dt`` against 0.66 at ``dt``.
            Broadband or non-stationary input favours a shorter membrane and
            slow stationary input a longer one; ``5 * dt`` is the value whose
            worst case over a sine, a two-tone, a chirp and filtered noise is
            highest. A membrane proportional to each channel's own scale
            :math:`\mu_k` is worse than a flat multiple, since the cascade
            already carries the scale covariance and a proportional membrane
            only adds scale-dependent delay.
        normalize (bool): Divide channels by their kernel norms. Defaults to
            True.
        kernel_length (Optional[int]): Impulse-response length in samples.
        learn_threshold (bool): Make thresholds learnable. Defaults to False.
    """

    def __init__(
        self,
        n_channels: int = 5,
        dt: float = 1e-3,
        mu_max: float = 0.2,
        c: float = 2.0,
        cascade_depth_max: int = 7,
        threshold: Optional[torch.Tensor] = None,
        mu_mem: Optional[torch.Tensor] = None,
        normalize: bool = True,
        kernel_length: Optional[int] = None,
        learn_threshold: bool = False,
    ):
        super().__init__()
        self.analysis = DifferenceOfTimeCausalKernels(
            n_channels=n_channels,
            dt=dt,
            mu_max=mu_max,
            c=c,
            cascade_depth_max=cascade_depth_max,
            kernel_length=kernel_length,
        )
        self.dt = dt
        self.normalize = normalize

        # The lowpass channel reuses the coarsest smoothing scale.
        mus = self.analysis.mus
        mus_per_channel = torch.cat([mus, mus[-1:]])
        depths = self.analysis.cascade_depths + [self.analysis.cascade_depths[-1]]

        if threshold is None:
            sqrt_depth = torch.tensor([float(d) ** 0.5 for d in depths])
            threshold = 10.0 * mus_per_channel * sqrt_depth
        threshold = torch.as_tensor(threshold, dtype=torch.float32).expand(n_channels)
        if learn_threshold:
            self.threshold = torch.nn.Parameter(threshold.clone())
        else:
            self.register_buffer("threshold", threshold.clone())

        if mu_mem is None:
            mu_mem = torch.full((n_channels,), float(MEMBRANE_STEPS * dt))
        self.register_buffer(
            "mu_mem", torch.as_tensor(mu_mem, dtype=torch.float32).expand(n_channels)
        )

        self._prescale_cache: dict = {}

    @property
    def n_channels(self) -> int:
        """Analysis channels; the spiking output has twice as many rows."""
        return self.analysis.n_channels

    def channel_prescale(self, length: Optional[int] = None) -> torch.Tensor:
        r"""Per-channel factor applied before thresholding, shape ``(n_channels,)``.

        The reciprocal of each channel's kernel L2 norm when ``normalize`` is
        set, and ones otherwise.
        """
        if not self.normalize:
            return torch.ones(self.n_channels)
        length = length or self.analysis.kernel_length or 4096
        if length not in self._prescale_cache:
            norms = bandpass_kernel_norms(self.analysis.kernels(length))
            self._prescale_cache[length] = 1.0 / (norms + 1e-10)
        return self._prescale_cache[length]

    def neurons(self) -> LIFBoxCell:
        r"""The threshold stage as one :class:`~norse.torch.LIFBoxCell`.

        Holds ``2 * n_channels`` neurons whose membranes decay by exactly
        :math:`e^{-\Delta t/\mu_{\text{mem}}}` per step and which reset to
        zero on firing.
        """
        paired_mu = self.mu_mem.repeat_interleave(2)
        paired_threshold = self.threshold.repeat_interleave(2)
        return LIFBoxCell(
            LIFBoxParameters(
                tau_mem_inv=exponential_tau_mem_inv(paired_mu, self.dt),
                v_leak=torch.zeros_like(paired_mu),
                v_th=paired_threshold,
                v_reset=torch.zeros_like(paired_mu),
            ),
            dt=self.dt,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Encode a signal into spikes, ``(T, *batch)`` -> ``(T, *batch, 2 * n_channels)``.

        Time first and neurons last, so the spike trains drop straight into a
        Norse pipeline as its encoding stage. Nothing here touches the
        decoding path, and the output carries gradients through the
        threshold's surrogate, so the encoder trains end to end.
        """
        length = input_tensor.shape[0]
        coefficients = _time_first_to_channels(self.analysis(input_tensor))
        prescale = self.channel_prescale(length).to(
            device=coefficients.device, dtype=coefficients.dtype
        )
        scaled = coefficients * prescale.reshape(-1, *([1] * (coefficients.dim() - 1)))
        # Polarity pairs: [+x_0, -x_0, +x_1, -x_1, ...]
        paired = torch.stack([scaled, -scaled], dim=1).flatten(0, 1)

        # The threshold parameters are per-neuron, so the neuron axis is last
        # while stepping, which is also the layout the result keeps.
        cell = self.neurons()
        spikes, state = [], None
        for n in range(length):
            z, state = cell(paired[..., n].movedim(0, -1), state)
            spikes.append(z)
        return torch.stack(spikes, dim=0)

    def decode(self, spikes: torch.Tensor) -> torch.Tensor:
        r"""Recover the analysis coefficients, ``(T, *batch, 2C)`` -> ``(T, *batch, C)``.

        Collapses each polarity pair into a signed train, inverts the
        membrane integration that produced the spikes, and undoes the
        prescaling applied before thresholding. The result is an estimate of
        what :meth:`DifferenceOfTimeCausalKernels.forward` produced, so it
        can be combined by the non-spiking transform's own
        :meth:`~DifferenceOfTimeCausalKernels.reconstruct`.
        """
        signed = (spikes[..., 0::2] - spikes[..., 1::2]).movedim(-1, 0).movedim(1, -1)
        length = signed.shape[-1]
        coefficients = invert_membrane_integration(
            signed, self.threshold, self.mu_mem, self.dt
        )
        prescale = self.channel_prescale(length).to(
            device=signed.device, dtype=signed.dtype
        )
        scaled = coefficients / prescale.reshape(-1, *([1] * (signed.dim() - 1)))
        return _channels_to_time_first(scaled)

    def reconstruct(self, spikes: torch.Tensor) -> torch.Tensor:
        r"""Decode spikes back to a signal, ``(T, *batch, 2C)`` -> ``(T, *batch)``.

        :meth:`decode` recovers the per-channel coefficients and the
        non-spiking transform telescopes them, so synthesis follows the same
        path as the exact inverse.

        The default weights are the unit weights of the exact telescoping
        sum, which need no calibration and no reference signal. Recovery is
        then approximate: it is quantised by the threshold, delayed by the
        membrane, and biased low in amplitude, because a spike discharges the
        membrane from :math:`v \ge \theta` rather than from exactly
        :math:`\theta`. Expect correlations around 0.7--0.9 against the
        input.

        :meth:`fit_oracle_weights` replaces the weights with a least-squares
        fit against a known signal, which reaches about 0.98 but requires
        that signal; see its caveats.
        """
        return self.analysis.reconstruct(self.decode(spikes))

    def fit_oracle_weights(self, signal: torch.Tensor) -> torch.Tensor:
        r"""Fit the reconstruction weights to a known signal, by least squares.

        Solves :math:`\min_b \lVert f - \sum_k b_k \hat x_k \rVert_2` over the
        decoded coefficients and stores the result as the non-spiking
        transform's ``channel_weights``.

        Warning:
            This is an *oracle* decoder, not a calibration. It needs the clean
            signal that the spikes came from, which a deployed encoder does
            not have, and the solution is specific to that signal rather than
            a fixed per-channel gain: fitting the same 4-channel encoder
            against a 3 Hz and then a 7 Hz sine gives
            ``[0.40, 11.1, 13.4, 4.7]`` and ``[1.28, 14.1, 7.1, 2.1]``, and
            some signals fit negative weights. Read it as an upper bound on
            what any linear decoder can recover from these spike trains, and
            as a diagnostic for how much of the signal survives encoding --
            not as a decoder to deploy.

        Args:
            signal: Shape ``(T,)``, the signal the spikes were encoded from.

        Returns:
            Shape ``(n_channels,)``, the fitted weights, also stored on
            ``self.analysis.channel_weights``.
        """
        if signal.dim() != 1:
            raise ValueError(f"signal must be 1-dimensional, got {signal.shape}")
        decoded = self.decode(self(signal)).detach()
        solution = torch.linalg.lstsq(
            decoded, signal.detach().unsqueeze(1)
        ).solution.squeeze(1)
        # reconstruct() computes w_K x_K - sum_{k<K} w_k x_k, so the
        # telescoping signs have to come back out of the fitted coefficients.
        weights = -solution
        weights[-1] = solution[-1]
        with torch.no_grad():
            self.analysis.channel_weights.copy_(weights.to(torch.float32))
        return weights

    def to_nir(self):
        """Export as a :class:`nir.NIRGraph` of ``2 * n_channels`` outputs."""
        from norse.torch.utils.export_nir_wavelet import from_spiking_wavelet

        return from_spiking_wavelet(self)

    def extra_repr(self) -> str:
        return f"n_channels={self.n_channels}, dt={self.dt}, normalize={self.normalize}"


class SpikingDifferenceOfExponentials(SpikingDifferenceOfTimeCausalKernels):
    r"""Spiking difference-of-exponentials (DoE) wavelet encoder.

    :class:`SpikingDifferenceOfTimeCausalKernels` with a single exponential
    per scale, the lightest causal variant.

    Parameters:
        n_channels (int): Analysis channels, at least 2.
        dt (float): Sample period in seconds.
        mu_max (float): Coarsest smoothing scale in seconds.
        c (float): Ratio between adjacent scales, ``c > 1``.
        threshold (Optional[torch.Tensor]): Firing threshold per channel.
        mu_mem (Optional[torch.Tensor]): Membrane time constant per channel.
            Defaults to ``5 * dt``.
        normalize (bool): Divide channels by their kernel norms.
        kernel_length (Optional[int]): Impulse-response length in samples.
        learn_threshold (bool): Make thresholds learnable.
    """

    def __init__(
        self,
        n_channels: int = 3,
        dt: float = 1e-3,
        mu_max: float = 0.2,
        c: float = 2.0**0.5,
        threshold: Optional[torch.Tensor] = None,
        mu_mem: Optional[torch.Tensor] = None,
        normalize: bool = True,
        kernel_length: Optional[int] = None,
        learn_threshold: bool = False,
    ):
        super().__init__(
            n_channels=n_channels,
            dt=dt,
            mu_max=mu_max,
            c=c,
            cascade_depth_max=1,
            threshold=threshold,
            mu_mem=mu_mem,
            normalize=normalize,
            kernel_length=kernel_length,
            learn_threshold=learn_threshold,
        )
