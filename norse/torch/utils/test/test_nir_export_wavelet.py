import math

import nir
import numpy as np
import pytest
import torch

from norse.torch.module.wavelet import (
    DifferenceOfExponentials,
    DifferenceOfTimeCausalKernels,
    SpikingDifferenceOfExponentials,
    SpikingDifferenceOfTimeCausalKernels,
)
from norse.torch.utils.export_nir_wavelet import (
    connectivity_matrix,
    passthrough_tau,
    polarity_rows,
)


def test_passthrough_tau_degenerates_in_float32():
    dt = 1e-3
    tau = passthrough_tau(dt)
    alpha = math.exp(-dt / tau)
    assert alpha < float(np.finfo(np.float32).tiny)
    assert np.float32(1.0) - np.float32(alpha) == np.float32(1.0)


def test_passthrough_tau_rejects_non_positive_dt():
    with pytest.raises(ValueError, match="dt must be positive"):
        passthrough_tau(0.0)


def test_connectivity_matrix_encodes_the_telescoping_structure():
    conn = connectivity_matrix(4)
    assert conn.shape == (4, 4)
    # Row 0 is L_1 - f, the only row touching the passthrough column.
    assert conn[0, 0] == 1 and conn[0, 3] == -1
    assert conn[1, 0] == -1 and conn[1, 1] == 1
    assert conn[2, 1] == -1 and conn[2, 2] == 1
    # The lowpass row taps the coarsest scale.
    assert conn[3, 2] == 1
    assert np.count_nonzero(conn[3]) == 1


def test_connectivity_matrix_applies_prescale():
    prescale = np.array([2.0, 3.0, 5.0])
    conn = connectivity_matrix(3, prescale=prescale)
    assert conn[0, 0] == 2.0 and conn[0, 2] == -2.0
    assert conn[1, 0] == -3.0 and conn[1, 1] == 3.0
    assert conn[2, 1] == 5.0


def test_connectivity_matrix_reproduces_dot_analysis():
    # Applying the matrix to [L_1..L_K, f] must give the analysis channels.
    n = 4
    conn = torch.as_tensor(connectivity_matrix(n), dtype=torch.float64)
    smoothed = torch.randn(n - 1, dtype=torch.float64)
    signal = torch.randn((), dtype=torch.float64)
    lanes = torch.cat([smoothed, signal.reshape(1)])
    got = conn @ lanes
    expected = torch.cat(
        [
            (smoothed[0] - signal).reshape(1),
            smoothed[1:] - smoothed[:-1],
            smoothed[-1:],
        ]
    )
    assert torch.allclose(got, expected, atol=1e-12)


def test_polarity_rows_interleaves_signs():
    conn = np.array([[1.0, -1.0], [2.0, 3.0]])
    out = polarity_rows(conn)
    assert out.shape == (4, 2)
    assert np.array_equal(out[0], conn[0])
    assert np.array_equal(out[1], -conn[0])
    assert np.array_equal(out[2], conn[1])
    assert np.array_equal(out[3], -conn[1])


def test_non_spiking_export_is_a_chain():
    w = DifferenceOfExponentials(n_channels=4, dt=1e-3, mu_max=0.02)
    graph = w.to_nir()
    assert isinstance(graph, nir.NIRGraph)
    assert isinstance(graph.nodes["input"], nir.Input)
    assert isinstance(graph.nodes["fanout"], nir.Affine)
    assert isinstance(graph.nodes["li_stage_0"], nir.LI)
    assert isinstance(graph.nodes["connectivity"], nir.Affine)
    assert isinstance(graph.nodes["output"], nir.Output)
    # Every node has at most one successor: a strict chain.
    sources = [a for a, _ in graph.edges]
    assert len(sources) == len(set(sources))


def test_non_spiking_export_has_no_lif():
    w = DifferenceOfExponentials(n_channels=3)
    assert not any(isinstance(n, nir.LIF) for n in w.to_nir().nodes.values())


def test_non_spiking_export_widths():
    n = 5
    w = DifferenceOfExponentials(n_channels=n, dt=1e-3, mu_max=0.02)
    graph = w.to_nir()
    assert graph.nodes["fanout"].weight.shape == (n, 1)  # K smoothing + passthrough
    assert graph.nodes["li_stage_0"].tau.shape == (n,)
    assert graph.nodes["connectivity"].weight.shape == (n, n)
    assert graph.nodes["output"].output_type["output"].tolist() == [n]


def test_export_carries_the_scale_time_constants():
    w = DifferenceOfExponentials(n_channels=4, dt=1e-3, mu_max=0.02)
    taus = w.to_nir().nodes["li_stage_0"].tau
    expected = [float(t[0]) for t in w.cascade_taus]
    assert np.allclose(taus[:-1], expected, rtol=1e-6)
    assert taus[-1] == pytest.approx(passthrough_tau(w.dt), rel=1e-6)


def test_deep_cascade_gets_one_li_stage_per_level():
    w = DifferenceOfTimeCausalKernels(
        n_channels=3, dt=1e-3, mu_max=0.5, cascade_depth_max=3
    )
    assert w.cascade_depths == [3, 3]
    graph = w.to_nir()
    stages = [k for k in graph.nodes if k.startswith("li_stage_")]
    assert len(stages) == 3


def test_export_rejects_unequal_cascade_depths():
    w = DifferenceOfTimeCausalKernels(
        n_channels=4, dt=1e-3, mu_max=0.05, cascade_depth_max=7
    )
    w.cascade_depths = [1, 2, 3]
    with pytest.raises(ValueError, match="equal cascade depths"):
        w.to_nir()


def test_export_rejects_deep_cascades():
    w = DifferenceOfTimeCausalKernels(n_channels=3, dt=1e-3, mu_max=0.05)
    w.cascade_depths = [10, 10]
    with pytest.raises(ValueError, match="depth >= 10"):
        w.to_nir()


def test_spiking_export_doubles_the_width():
    n = 3
    w = SpikingDifferenceOfExponentials(n_channels=n, dt=1e-3, mu_max=0.02)
    graph = w.to_nir()
    assert isinstance(graph.nodes["lif"], nir.LIF)
    assert graph.nodes["connectivity"].weight.shape == (2 * n, n)
    assert graph.nodes["lif"].tau.shape == (2 * n,)
    assert graph.nodes["output"].output_type["output"].tolist() == [2 * n]


def test_spiking_export_pairs_are_opposite():
    w = SpikingDifferenceOfExponentials(n_channels=3, dt=1e-3, mu_max=0.02)
    weight = w.to_nir().nodes["connectivity"].weight
    assert np.allclose(weight[0::2], -weight[1::2])


def test_spiking_export_repeats_thresholds_per_pair():
    w = SpikingDifferenceOfExponentials(n_channels=3, dt=1e-3, mu_max=0.02)
    lif = w.to_nir().nodes["lif"]
    assert np.allclose(lif.v_threshold[0::2], lif.v_threshold[1::2])
    assert np.allclose(lif.v_threshold[0::2], w.threshold.numpy(), rtol=1e-6)
    assert np.allclose(lif.v_reset, 0.0)
    assert np.allclose(lif.tau[0::2], w.mu_mem.numpy(), rtol=1e-6)


def test_spiking_export_folds_prescale_into_connectivity():
    plain = SpikingDifferenceOfExponentials(
        n_channels=3, dt=1e-3, mu_max=0.02, normalize=False
    )
    scaled = SpikingDifferenceOfExponentials(
        n_channels=3, dt=1e-3, mu_max=0.02, normalize=True
    )
    w_plain = plain.to_nir().nodes["connectivity"].weight
    w_scaled = scaled.to_nir().nodes["connectivity"].weight
    assert np.allclose(np.abs(w_plain).max(), 1.0)
    assert not np.allclose(w_plain, w_scaled)


def test_exported_graph_survives_nir_serialisation(tmp_path):
    w = SpikingDifferenceOfExponentials(n_channels=3, dt=1e-3, mu_max=0.02)
    path = tmp_path / "spiking_doe.nir"
    nir.write(str(path), w.to_nir())
    loaded = nir.read(str(path))
    assert set(loaded.nodes) == set(w.to_nir().nodes)


def test_imported_passthrough_lane_is_stable():
    # Reimporting the export must not turn the degenerate lane into a
    # divergent integrator.
    from norse.torch.utils.import_nir import from_nir

    w = DifferenceOfExponentials(n_channels=3, dt=1e-3, mu_max=0.02)
    module = from_nir(w.to_nir(), dt=w.dt)
    out = module(torch.ones(1))
    for _ in range(50):
        out = module(torch.ones(1))
        value = out[0] if isinstance(out, tuple) else out
        assert torch.all(torch.isfinite(torch.as_tensor(value)))
        assert torch.all(torch.as_tensor(value).abs() < 1e3)
