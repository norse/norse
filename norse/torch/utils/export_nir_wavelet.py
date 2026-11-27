r"""NIR export for the time-causal wavelet transforms.

Both the non-spiking and spiking variants export as a single linear chain
ending at one :class:`nir.Output`, which is what most neuromorphic backends
ingest:

.. code-block:: text

    non-spiking: Input -> fanout Affine -> LI bank -> connectivity Affine -> Output
    spiking:     Input -> fanout Affine -> LI bank -> connectivity Affine -> LIF -> Output

The LI bank carries ``K + 1`` lanes for ``K`` smoothing scales. The extra
lane reproduces the raw signal :math:`f`, which the leading bandpass channel
:math:`\Delta L_1 = L_1 - f` needs. An LI can only emit smoothed versions of
its input, so the lane is given a time constant small enough that
:math:`\alpha = e^{-\Delta t/\tau}` underflows and the input factor
:math:`1 - \alpha` is exactly one, degenerating it to :math:`v[n] = x[n]`.
:func:`passthrough_tau` asserts both conditions in float32 so a bad ``dt``
fails at export rather than silently adding a smoothing scale.

The connectivity Affine maps the bank's ``K + 1`` lanes onto the output
channels: row 0 is :math:`L_1 - f` (the only row touching the passthrough
lane), rows :math:`1 \dots K-1` are forward differences
:math:`L_k - L_{k-1}`, and row :math:`K` taps the coarsest scale.

A :class:`nir.LIF` fires only on positive threshold crossings while every
analysis channel is signed, so the spiking export widens each channel into a
polarity pair driven by :math:`+x_k` and :math:`-x_k`, stacked as
``[pos_0, neg_0, pos_1, neg_1, ...]``. The spiking graph is therefore
``2 * n_channels`` wide.
"""

import math
from typing import List, Optional, Sequence, Tuple

import nir
import numpy as np

MAX_CASCADE_DEPTH = 10

# tau of the passthrough lane as a fraction of dt: alpha = exp(-100), far
# below the float32 normal floor, and well under any plausible backend floor.
PASSTHROUGH_TAU_FRACTION = 1e-2


def passthrough_tau(dt: float) -> float:
    r"""Time constant of the raw-signal lane.

    The lane passes its input through only if :math:`\alpha = e^{-\Delta
    t/\tau}` is negligible *and* :math:`1 - \alpha` is exactly one in the
    consumer's arithmetic. Both are checked in float32, the narrowest format
    a backend is likely to use.

    Args:
        dt: Sample period in seconds.

    Returns:
        The lane's time constant in seconds.

    Raises:
        ValueError: If ``dt`` is not positive, or the lane would not
            degenerate to a passthrough in float32.
    """
    dt = float(dt)
    if not dt > 0:
        raise ValueError(f"dt must be positive, got {dt}")
    tau = dt * PASSTHROUGH_TAU_FRACTION
    alpha = math.exp(-dt / tau)
    tiny = float(np.finfo(np.float32).tiny)
    if not alpha < tiny:
        raise ValueError(
            f"passthrough lane does not degenerate: alpha={alpha:g} is not "
            f"below the float32 normal floor {tiny:g}"
        )
    if np.float32(1.0) - np.float32(alpha) != np.float32(1.0):
        raise ValueError(
            f"passthrough lane input factor (1 - alpha) != 1 in float32 "
            f"(alpha={alpha:g})"
        )
    return tau


def _validate(cascade_depths: Sequence[int]) -> None:
    depths = list(cascade_depths)
    if max(depths) >= MAX_CASCADE_DEPTH:
        raise ValueError(
            f"NIR export rejects cascades of depth >= {MAX_CASCADE_DEPTH}, "
            f"whose stage time constants are not numerically representable on "
            f"most neuromorphic targets; got cascade_depths={depths}"
        )
    if len(set(depths)) > 1:
        raise ValueError(
            f"NIR export requires equal cascade depths across channels, since "
            f"each stage becomes one multi-channel LI bank; got "
            f"cascade_depths={depths}"
        )


def connectivity_matrix(
    n_channels: int, prescale: Optional[np.ndarray] = None
) -> np.ndarray:
    r"""The ``(n_channels, K + 1)`` map from LI lanes to output channels.

    Columns :math:`0 \dots K-1` are the smoothing scales and column
    :math:`K` is the passthrough lane carrying :math:`f`.

    Args:
        n_channels: Number of output channels, ``K + 1``.
        prescale: Optional per-channel factor folded into the row magnitudes.

    Returns:
        Array of shape ``(n_channels, n_channels)``.
    """
    k_max = n_channels - 1
    if prescale is None:
        prescale = np.ones(n_channels)
    conn = np.zeros((n_channels, k_max + 1), dtype=np.float32)
    conn[0, 0] = float(prescale[0])
    conn[0, k_max] = -float(prescale[0])
    for k in range(1, k_max):
        conn[k, k - 1] = -float(prescale[k])
        conn[k, k] = float(prescale[k])
    conn[k_max, k_max - 1] = float(prescale[k_max])
    return conn


def polarity_rows(conn: np.ndarray) -> np.ndarray:
    """Interleave rows into polarity pairs ``[+row_0, -row_0, +row_1, ...]``."""
    out = np.empty((2 * conn.shape[0], conn.shape[1]), dtype=conn.dtype)
    out[0::2] = conn
    out[1::2] = -conn
    return out


def _per_pair(values: Sequence[float]) -> np.ndarray:
    """Repeat each per-channel value once per neuron of its polarity pair."""
    return np.repeat(np.asarray(values, dtype=np.float32), 2)


def _li_bank(
    cascade_taus: Sequence[np.ndarray], dt: float, n_channels: int, source: str
) -> Tuple[str, dict, List[Tuple[str, str]]]:
    """Fanout plus one multi-channel LI node per cascade stage."""
    k_max = n_channels - 1
    depth = len(cascade_taus[0])
    width = k_max + 1
    tau_pass = passthrough_tau(dt)

    nodes = {
        "fanout": nir.Affine(
            weight=np.ones((width, 1), dtype=np.float32),
            bias=np.zeros(width, dtype=np.float32),
        )
    }
    edges = [(source, "fanout")]

    previous = "fanout"
    for stage in range(depth):
        name = f"li_stage_{stage}"
        stage_taus = np.array(
            [float(cascade_taus[k][stage]) for k in range(k_max)] + [tau_pass],
            dtype=np.float32,
        )
        nodes[name] = nir.LI(
            tau=stage_taus,
            r=np.ones(width, dtype=np.float32),
            v_leak=np.zeros(width, dtype=np.float32),
        )
        edges.append((previous, name))
        previous = name
    return previous, nodes, edges


def from_wavelet(wavelet) -> nir.NIRGraph:
    """Export a non-spiking wavelet transform as a :class:`nir.NIRGraph`.

    Args:
        wavelet: A :class:`~norse.torch.module.wavelet.DifferenceOfTimeCausalKernels`
            or subclass.

    Returns:
        A graph of ``n_channels`` outputs carrying the bandpass differences
        and the lowpass residual, in the order
        :meth:`~norse.torch.module.wavelet.DifferenceOfTimeCausalKernels.forward`
        produces them.
    """
    _validate(wavelet.cascade_depths)
    taus = [t.numpy() for t in wavelet.cascade_taus]

    nodes = {"input": nir.Input(input_type={"input": np.array([1])})}
    edges: List[Tuple[str, str]] = []

    terminal, li_nodes, li_edges = _li_bank(
        taus, wavelet.dt, wavelet.n_channels, "input"
    )
    nodes.update(li_nodes)
    edges.extend(li_edges)

    nodes["connectivity"] = nir.Affine(
        weight=connectivity_matrix(wavelet.n_channels),
        bias=np.zeros(wavelet.n_channels, dtype=np.float32),
    )
    edges.append((terminal, "connectivity"))

    nodes["output"] = nir.Output(output_type={"output": np.array([wavelet.n_channels])})
    edges.append(("connectivity", "output"))

    return nir.NIRGraph(nodes=nodes, edges=edges, type_check=False)


def from_spiking_wavelet(wavelet) -> nir.NIRGraph:
    """Export a spiking wavelet transform as a :class:`nir.NIRGraph`.

    Args:
        wavelet: A :class:`~norse.torch.module.wavelet.SpikingDifferenceOfTimeCausalKernels`
            or subclass.

    Returns:
        A graph of ``2 * n_channels`` spiking outputs, ordered
        ``[pos_0, neg_0, pos_1, neg_1, ...]``.
    """
    analysis = wavelet.analysis
    _validate(analysis.cascade_depths)
    taus = [t.numpy() for t in analysis.cascade_taus]

    n_out = 2 * analysis.n_channels
    nodes = {"input": nir.Input(input_type={"input": np.array([1])})}
    edges: List[Tuple[str, str]] = []

    terminal, li_nodes, li_edges = _li_bank(
        taus, analysis.dt, analysis.n_channels, "input"
    )
    nodes.update(li_nodes)
    edges.extend(li_edges)

    conn = polarity_rows(
        connectivity_matrix(
            analysis.n_channels, prescale=wavelet.channel_prescale().numpy()
        )
    )
    nodes["connectivity"] = nir.Affine(
        weight=conn, bias=np.zeros(n_out, dtype=np.float32)
    )
    edges.append((terminal, "connectivity"))

    nodes["lif"] = nir.LIF(
        tau=_per_pair(wavelet.mu_mem.detach().numpy()),
        r=np.ones(n_out, dtype=np.float32),
        v_leak=np.zeros(n_out, dtype=np.float32),
        v_threshold=_per_pair(wavelet.threshold.detach().numpy()),
        v_reset=np.zeros(n_out, dtype=np.float32),
    )
    edges.append(("connectivity", "lif"))

    nodes["output"] = nir.Output(output_type={"output": np.array([n_out])})
    edges.append(("lif", "output"))

    return nir.NIRGraph(nodes=nodes, edges=edges, type_check=False)
