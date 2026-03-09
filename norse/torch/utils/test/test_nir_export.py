import torch
import nir
import numpy as np

import norse.torch as norse

import pytest


def test_conv2d():
    m = norse.SequentialState(torch.nn.Conv2d(1, 2, 3))
    # type_check=False because Conv2d is created without input_shape
    graph = norse.to_nir(m, type_check=False)
    assert len(graph.nodes) == 3
    assert isinstance(graph.nodes["input_tensor"], nir.Input)
    assert isinstance(graph.nodes["_0"], nir.Conv2d)
    assert isinstance(graph.nodes["output"], nir.Output)
    assert len(graph.edges) == 2


def test_sequential():
    m = norse.SequentialState(
        norse.LIFBoxCell(),
        torch.nn.Linear(10, 2),
        norse.LIBoxCell(),
        torch.nn.Linear(2, 1),
    )
    # type_check=False because neuron models lack shape information
    graph = norse.to_nir(m, type_check=False)
    assert len(graph.nodes) == 6  # 4 + 2 for input and output
    assert isinstance(graph.nodes["input_tensor"], nir.Input)
    assert isinstance(graph.nodes["_0"], nir.LIF)
    assert isinstance(graph.nodes["_1"], nir.Affine)
    assert isinstance(graph.nodes["_2"], nir.LI)
    assert isinstance(graph.nodes["_3"], nir.Affine)
    assert isinstance(graph.nodes["output"], nir.Output)
    assert len(graph.edges) == 5


def test_torch_subclass():
    class TorchSubclassModel(torch.nn.Module):
        def __init__(self):
            super(TorchSubclassModel, self).__init__()

            self.lif0 = norse.LIFBoxCell()
            self.l0 = torch.nn.Linear(10, 2)
            self.lif1 = norse.LIBoxCell()
            self.l1 = torch.nn.Linear(2, 1)

            self.states = [None, None]

        def forward(self, x):
            z, self.states[0] = self.lif0(x, self.states[0])
            z = self.l0(z)
            z, self.states[1] = self.lif1(x, self.states[1])
            z = self.l1(z)
            return z

    m = TorchSubclassModel()
    graph = norse.to_nir(m, type_check=False)
    assert len(graph.nodes) == 6  # 4 + 2 for input and output
    assert isinstance(graph.nodes["x"], nir.Input)
    assert isinstance(graph.nodes["lif0"], nir.LIF)
    assert isinstance(graph.nodes["l0"], nir.Affine)
    assert isinstance(graph.nodes["lif1"], nir.LI)
    assert isinstance(graph.nodes["l1"], nir.Affine)
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
    assert np.allclose(node.tau, 1.0 / p.tau_mem_inv.numpy())
    assert isinstance(node.tau, np.ndarray)
    assert np.allclose(node.v_leak, p.v_leak.numpy())
    assert isinstance(node.v_leak, np.ndarray)
    node = norse.to_nir(m, time_scaling_factor=0.5)
    assert np.allclose(node.tau, 0.5 / p.tau_mem_inv.numpy())
    assert isinstance(node.tau, np.ndarray)
    assert np.allclose(node.v_leak, p.v_leak.numpy())
    assert isinstance(node.v_leak, np.ndarray)


def test_lif_sequential():
    network = norse.SequentialState(norse.LIFCell(), torch.nn.Linear(1, 1))
    # type_check=False because LIFCell has no inherent shape information
    # (neurons are element-wise and work with any input size)
    graph = norse.to_nir(network, type_check=False)
    assert len(graph.nodes) == 4  # 2 + 2 for input and output
    assert isinstance(graph.nodes["input_tensor"], nir.Input)
    assert isinstance(graph.nodes["_0"], nir.CubaLIF)
    assert isinstance(graph.nodes["_1"], nir.Affine)
    assert isinstance(graph.nodes["output"], nir.Output)
    assert len(graph.edges) == 3


def test_lif_varying_time_scaling_factor():
    p = norse.LIFParameters(
        tau_mem_inv=torch.tensor([100.0]),
        tau_syn_inv=torch.tensor([100.0]),
        v_leak=torch.tensor([0.0]),
    )
    m = norse.LIFCell(p)
    node = norse.to_nir(m, time_scaling_factor=1.0)
    assert isinstance(node, nir.CubaLIF)
    assert np.allclose(node.tau_mem, 1.0 / p.tau_mem_inv.numpy())
    assert isinstance(node.tau_mem, np.ndarray)
    assert np.allclose(node.tau_syn, 1.0 / p.tau_syn_inv.numpy())
    assert isinstance(node.tau_syn, np.ndarray)
    assert np.allclose(node.v_leak, p.v_leak.numpy())
    assert isinstance(node.v_leak, np.ndarray)
    node = norse.to_nir(m, time_scaling_factor=0.5)
    assert np.allclose(node.tau_mem, 0.5 / p.tau_mem_inv.numpy())
    assert isinstance(node.tau_mem, np.ndarray)
    assert np.allclose(node.tau_syn, 0.5 / p.tau_syn_inv.numpy())
    assert isinstance(node.tau_syn, np.ndarray)
    assert np.allclose(node.v_leak, p.v_leak.numpy())
    assert isinstance(node.v_leak, np.ndarray)


def test_lif_box_varying_time_scaling_factor():
    p = norse.LIFBoxParameters(
        tau_mem_inv=torch.tensor([100.0]), v_leak=torch.tensor([0.0])
    )
    m = norse.LIFBoxCell(p)
    node = norse.to_nir(m, time_scaling_factor=1.0)
    assert isinstance(node, nir.LIF)
    assert np.allclose(node.tau, 1.0 / p.tau_mem_inv.numpy())
    assert isinstance(node.tau, np.ndarray)
    assert np.allclose(node.v_leak, p.v_leak.numpy())
    assert isinstance(node.v_leak, np.ndarray)
    node = norse.to_nir(m, time_scaling_factor=0.5)
    assert np.allclose(node.tau, 0.5 / p.tau_mem_inv.numpy())
    assert isinstance(node.tau, np.ndarray)
    assert np.allclose(node.v_leak, p.v_leak.numpy())
    assert isinstance(node.v_leak, np.ndarray)


def test_lif_box_v_reset():
    p = norse.LIFBoxParameters(
        tau_mem_inv=torch.tensor([100.0, 200.0]),
        v_leak=torch.tensor([0.0, 0.0]),
        v_reset=torch.tensor([0.0, 0.0]),
    )
    m = norse.LIFBoxCell(p)
    node = norse.to_nir(m, time_scaling_factor=1.0)
    assert isinstance(node, nir.LIF)
    assert np.allclose(node.tau, 1.0 / p.tau_mem_inv.numpy())
    assert isinstance(node.tau, np.ndarray)
    assert np.allclose(node.v_leak, p.v_leak.numpy())
    assert isinstance(node.v_leak, np.ndarray)
    assert np.allclose(node.v_reset, p.v_reset.numpy())
    assert isinstance(node.v_reset, np.ndarray)


def test_lif_box_v_reset_default():
    p = norse.LIFBoxParameters(
        tau_mem_inv=torch.tensor([100.0, 200.0]), v_leak=torch.tensor([0.0, 0.0])
    )
    m = norse.LIFBoxCell(p)
    node = norse.to_nir(m, time_scaling_factor=1.0)
    assert isinstance(node, nir.LIF)
    assert np.allclose(node.tau, 1.0 / p.tau_mem_inv.numpy())
    assert isinstance(node.tau, np.ndarray)
    assert np.allclose(node.v_leak, p.v_leak.numpy())
    assert isinstance(node.v_leak, np.ndarray)
    assert np.allclose(node.v_reset, np.zeros_like(p.v_leak.numpy()))
    assert isinstance(node.v_reset, np.ndarray)


def test_custom_module():
    # Create custom module and define how it is mapped to NIR primitives
    class CustomModule(torch.nn.Module):
        def forward(self, x):
            return x

    def _map_custom_neuron(_: CustomModule):
        return nir.LIF(
            tau=np.array(1.0),
            r=np.array(1.0),
            v_leak=np.array(0.0),
            v_threshold=np.array(1.0),
            v_reset=np.array(0.0)
        )

    custom_mapping = { CustomModule: _map_custom_neuron }

    m = norse.SequentialState(
        CustomModule(),
        torch.nn.Linear(10, 2),
        norse.LIBoxCell(),
        torch.nn.Linear(2, 1),
    )

    # type_check=False because neuron models lack shape information
    graph = norse.to_nir(m, type_check=False, custom_mapping=custom_mapping)
    assert len(graph.nodes) == 6  # 4 + 2 for input and output
    assert isinstance(graph.nodes["input_tensor"], nir.Input)
    assert isinstance(graph.nodes["_0"], nir.LIF)
    assert isinstance(graph.nodes["_1"], nir.Affine)
    assert isinstance(graph.nodes["_2"], nir.LI)
    assert isinstance(graph.nodes["_3"], nir.Affine)
    assert isinstance(graph.nodes["output"], nir.Output)
    assert len(graph.edges) == 5


def test_custom_module_stateful():
    # Create custom stateful module and define how it is mapped to NIR primitives
    class CustomStatefulModule(torch.nn.Module):
        def forward(self, x, state):
            return x, state

    def _map_custom_neuron(_: CustomStatefulModule):
        return nir.LIF(
            tau=np.array(1.0),
            r=np.array(1.0),
            v_leak=np.array(0.0),
            v_threshold=np.array(1.0),
            v_reset=np.array(0.0)
        )

    custom_stateful_modules = { CustomStatefulModule }
    custom_mapping = { CustomStatefulModule: _map_custom_neuron }

    m = norse.SequentialState(
        CustomStatefulModule(),
        torch.nn.Linear(10, 2),
        norse.LIBoxCell(),
        torch.nn.Linear(2, 1),
    )

    # type_check=False because neuron models lack shape information
    graph = norse.to_nir(
        m, type_check=False,
        custom_stateful_modules=custom_stateful_modules,
        custom_mapping=custom_mapping
    )
    assert len(graph.nodes) == 6  # 4 + 2 for input and output
    assert isinstance(graph.nodes["input_tensor"], nir.Input)
    assert isinstance(graph.nodes["_0"], nir.LIF)
    assert isinstance(graph.nodes["_1"], nir.Affine)
    assert isinstance(graph.nodes["_2"], nir.LI)
    assert isinstance(graph.nodes["_3"], nir.Affine)
    assert isinstance(graph.nodes["output"], nir.Output)
    assert len(graph.edges) == 5


def test_bypass_module():
    m = norse.SequentialState(
        norse.LIFBoxCell(),
        torch.nn.Linear(10, 2),
        torch.nn.Dropout(),
        norse.LIBoxCell(),
        torch.nn.Linear(2, 1),
    )

    def map_none(_):
        return None

    bypass_map = {torch.nn.Dropout: map_none}

    graph = norse.to_nir(m, type_check=False, custom_mapping=bypass_map)
    assert len(graph.nodes) == 6  # 4 + 2 for input and output
    assert isinstance(graph.nodes["input_tensor"], nir.Input)
    assert isinstance(graph.nodes["_0"], nir.LIF)
    assert isinstance(graph.nodes["_1"], nir.Affine)
    assert "_2" not in graph.nodes  # 'Dropout' not present because it has been bypassed
    assert isinstance(graph.nodes["_3"], nir.LI)
    assert isinstance(graph.nodes["_4"], nir.Affine)
    assert isinstance(graph.nodes["output"], nir.Output)
    assert len(graph.edges) == 5
