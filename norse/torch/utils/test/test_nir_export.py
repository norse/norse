import torch
import nir

import norse.torch as norse


def test_conv2d():
    m = norse.SequentialState(torch.nn.Conv2d(1, 2, 3))
    graph = norse.to_nir(m, torch.randn(1, 1, 10, 10))
    assert len(graph.nodes) == 3
    assert isinstance(graph.nodes["input"], nir.Input)
    assert isinstance(graph.nodes["0"], nir.Conv2d)
    assert isinstance(graph.nodes["output"], nir.Output)
    assert len(graph.edges) == 2


def test_sequential():
    m = norse.SequentialState(
        norse.LIFBoxCell(),
        torch.nn.Linear(10, 2),
        norse.LIBoxCell(),
        torch.nn.Linear(2, 1),
    )
    graph = norse.to_nir(m, torch.randn(1, 10))
    assert len(graph.nodes) == 6  # 4 + 2 for input and output
    assert isinstance(graph.nodes["input"], nir.Input)
    assert isinstance(graph.nodes["0"], nir.LIF)
    assert isinstance(graph.nodes["1"], nir.Affine)
    assert isinstance(graph.nodes["2"], nir.LI)
    assert isinstance(graph.nodes["3"], nir.Affine)
    assert isinstance(graph.nodes["output"], nir.Output)
    assert len(graph.edges) == 5


def test_linear():
    in_features = 2
    out_features = 3
    m = torch.nn.Linear(in_features, out_features, bias=False)
    m2 = torch.nn.Linear(in_features, out_features, bias=True)
    graph = norse.to_nir(m, torch.randn(1, in_features))
    assert len(graph.nodes) == 3
    assert graph.nodes["norse"].weight.shape == (out_features, in_features)
    assert graph.nodes["norse"].bias.shape == m2.bias.shape


def test_li_dt_varying_dt():
    p = norse.LIBoxParameters(
        tau_mem_inv=torch.tensor([900.0]), v_leak=torch.tensor([0.0])
    )
    m = norse.LIBoxCell(p)
    graph = norse.to_nir(m, torch.randn(1, 1), dt=1.0)
    assert len(graph.nodes) == 3
    assert torch.allclose(graph.nodes["norse"].tau, 1.0 / p.tau_mem_inv)
    assert torch.allclose(graph.nodes["norse"].v_leak, p.v_leak)
    graph = norse.to_nir(m, torch.randn(1, 1), dt=0.5)
    assert len(graph.nodes) == 3
    assert torch.allclose(graph.nodes["norse"].tau, 0.5 / p.tau_mem_inv)
    assert torch.allclose(graph.nodes["norse"].v_leak, p.v_leak)


def test_lif_varying_dt():
    p = norse.LIFParameters(
        tau_mem_inv=torch.tensor([100.0]),
        tau_syn_inv=torch.tensor([100.0]),
        v_leak=torch.tensor([0.0]),
    )
    m = norse.LIFCell(p)
    graph = norse.to_nir(m, torch.randn(1, 1), dt=1.0)
    assert len(graph.nodes) == 3
    assert torch.allclose(graph.nodes["norse"].tau_mem, 1.0 / p.tau_mem_inv)
    assert torch.allclose(graph.nodes["norse"].tau_syn, 1.0 / p.tau_syn_inv)
    assert torch.allclose(graph.nodes["norse"].v_leak, p.v_leak)
    graph = norse.to_nir(m, torch.randn(1, 1), dt=0.5)
    assert len(graph.nodes) == 3
    assert torch.allclose(graph.nodes["norse"].tau_mem, 0.5 / p.tau_mem_inv)
    assert torch.allclose(graph.nodes["norse"].tau_syn, 0.5 / p.tau_syn_inv)
    assert torch.allclose(graph.nodes["norse"].v_leak, p.v_leak)


def test_lif_box_varying_dt():
    p = norse.LIFBoxParameters(
        tau_mem_inv=torch.tensor([100.0]), v_leak=torch.tensor([0.0])
    )
    m = norse.LIFBoxCell(p)
    graph = norse.to_nir(m, torch.randn(1, 1), dt=1.0)
    assert len(graph.nodes) == 3
    assert torch.allclose(graph.nodes["norse"].tau, 1.0 / p.tau_mem_inv)
    assert torch.allclose(graph.nodes["norse"].v_leak, p.v_leak)
    graph = norse.to_nir(m, torch.randn(1, 1), dt=0.5)
    assert len(graph.nodes) == 3
    assert torch.allclose(graph.nodes["norse"].tau, 0.5 / p.tau_mem_inv)
    assert torch.allclose(graph.nodes["norse"].v_leak, p.v_leak)
