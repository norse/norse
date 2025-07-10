import torch
import nir

import norse.torch as norse

import pytest


def test_conv2d():
    m = norse.SequentialState(torch.nn.Conv2d(1, 2, 3))
    graph = norse.to_nir(m, type_check=False)
    assert len(graph.nodes) == 3
    assert isinstance(graph.nodes["input_tensor"], nir.Input)
    assert isinstance(graph.nodes["_0"], nir.Conv2d)
    assert isinstance(graph.nodes["output"], nir.Output)
    assert len(graph.edges) == 2


@pytest.mark.skip("TODO: Fix assignment during symbolic tracing")
def test_sequential():
    m = norse.SequentialState(
        norse.LIFBoxCell(),
        torch.nn.Linear(10, 2),
        norse.LIBoxCell(),
        torch.nn.Linear(2, 1),
    )
    graph = norse.to_nir(m, type_check=False)
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
    m = torch.nn.Linear(in_features, out_features, bias=True)
    m2 = torch.nn.Linear(in_features, out_features, bias=True)
    node = norse.to_nir(m)
    assert isinstance(node, nir.Affine)
    assert node.weight.shape == (out_features, in_features)
    assert node.bias.shape == m2.bias.shape


def test_li_varying_time_scaling_factor():
    p = norse.LIBoxParameters(
        tau_mem_inv=torch.tensor([900.0]), v_leak=torch.tensor([0.0])
    )
    m = norse.LIBoxCell(p)
    node = norse.to_nir(m, time_scaling_factor=1.0)
    assert isinstance(node, nir.LI)
    assert torch.allclose(node.tau, 1.0 / p.tau_mem_inv)
    assert torch.allclose(node.v_leak, p.v_leak)
    node = norse.to_nir(m, time_scaling_factor=0.5)
    assert torch.allclose(node.tau, 0.5 / p.tau_mem_inv)
    assert torch.allclose(node.v_leak, p.v_leak)


def test_lif_varying_time_scaling_factor():
    p = norse.LIFParameters(
        tau_mem_inv=torch.tensor([100.0]),
        tau_syn_inv=torch.tensor([100.0]),
        v_leak=torch.tensor([0.0]),
    )
    m = norse.LIFCell(p)
    node = norse.to_nir(m, time_scaling_factor=1.0)
    assert isinstance(node, nir.CubaLIF)
    assert torch.allclose(node.tau_mem, 1.0 / p.tau_mem_inv)
    assert torch.allclose(node.tau_syn, 1.0 / p.tau_syn_inv)
    assert torch.allclose(node.v_leak, p.v_leak)
    node = norse.to_nir(m, time_scaling_factor=0.5)
    assert torch.allclose(node.tau_mem, 0.5 / p.tau_mem_inv)
    assert torch.allclose(node.tau_syn, 0.5 / p.tau_syn_inv)
    assert torch.allclose(node.v_leak, p.v_leak)


def test_lif_box_varying_time_scaling_factor():
    p = norse.LIFBoxParameters(
        tau_mem_inv=torch.tensor([100.0]), v_leak=torch.tensor([0.0])
    )
    m = norse.LIFBoxCell(p)
    node = norse.to_nir(m, time_scaling_factor=1.0)
    assert isinstance(node, nir.LIF)
    assert torch.allclose(node.tau, 1.0 / p.tau_mem_inv)
    assert torch.allclose(node.v_leak, p.v_leak)
    node = norse.to_nir(m, time_scaling_factor=0.5)
    assert torch.allclose(node.tau, 0.5 / p.tau_mem_inv)
    assert torch.allclose(node.v_leak, p.v_leak)


def test_lif_box_v_reset():
    p = norse.LIFBoxParameters(
        tau_mem_inv=torch.tensor([100.0, 200.0]),
        v_leak=torch.tensor([0.0, 0.0]),
        v_reset=torch.tensor([0.0, 0.0]),
    )
    m = norse.LIFBoxCell(p)
    node = norse.to_nir(m, time_scaling_factor=1.0)
    assert isinstance(node, nir.LIF)
    assert torch.allclose(node.tau, 1.0 / p.tau_mem_inv)
    assert torch.allclose(node.v_leak, p.v_leak)
    assert torch.allclose(node.v_reset, p.v_reset)


def test_lif_box_v_reset_default():
    p = norse.LIFBoxParameters(
        tau_mem_inv=torch.tensor([100.0, 200.0]), v_leak=torch.tensor([0.0, 0.0])
    )
    m = norse.LIFBoxCell(p)
    node = norse.to_nir(m, time_scaling_factor=1.0)
    assert isinstance(node, nir.LIF)
    assert torch.allclose(node.tau, 1.0 / p.tau_mem_inv)
    assert torch.allclose(node.v_leak, p.v_leak)
    assert torch.allclose(node.v_reset, torch.zeros_like(p.v_leak))
