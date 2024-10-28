from dataclasses import fields

import pytest, platform
import torch

from norse.torch.functional.lif import LIFState, LIFParameters
from norse.torch.module.lif import (
    LIFCell,
    LIF,
)


class SNNetwork(torch.nn.Module):
    def __init__(self):
        super(SNNetwork, self).__init__()
        self.l0 = LIF()
        self.l1 = LIF()
        self.s0 = self.s1 = None

    def forward(self, spikes):
        spikes, self.s0 = self.l0(spikes, self.s0)
        _, self.s1 = self.l1(spikes, self.s1)
        return self.s1.v.squeeze()


def test_lif_cell_feedforward():
    cell = LIFCell()
    data = torch.randn(5, 2)
    out, s = cell(data)

    for x in s:
        assert x.shape == (5, 2)
    assert out.shape == (5, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not supported")
def test_lif_cell_feedforward_cuda():
    cell = LIFCell(LIFParameters().cuda()).cuda()
    data = torch.randn(5, 2).cuda()
    out, s = cell(data)
    assert out.device.type == "cuda"

    for x in s:
        assert x.shape == (5, 2)
    assert out.shape == (5, 2)


@pytest.mark.skipif(
    (
        True  # TODO: This crashes on 2.4 for some reason...
        or not platform.system() == "Linux"
    ),
    reason="Only Linux supports torch.compile",
)
def test_lif_cell_feedforward_compile():
    layer = LIFCell()
    layer = torch.compile(layer, fullgraph=True)
    data = torch.randn(5, 4)
    out, s = layer(data)
    assert out.shape == (5, 4)
    for x in s:
        assert x.shape == (5, 4)


def test_lif_in_time():
    layer = LIF()
    data = torch.randn(10, 5, 2)
    out, _ = layer(data)

    assert out.shape == (10, 5, 2)


def test_lif_feedforward_cell_backward():
    # Tests that gradient variables can be used in subsequent applications
    cell = LIFCell()
    data = torch.randn(5, 4)
    out, s = cell(data)
    out, _ = cell(out, s)
    loss = out.sum()
    loss.backward()


def test_lif_feedforward_layer():
    layer = LIF()
    data = torch.randn(10, 5, 4)
    out, s = layer(data)
    assert out.shape == (10, 5, 4)
    for x in s:
        assert x.shape == (5, 4)


@pytest.mark.skipif(
    (
        True  # TODO: This crashes on 2.4 for some reason...
        or not platform.system() == "Linux"
    ),
    reason="Only Linux supports torch.compile",
)
def test_lif_feedforward_layer_compile():
    layer = LIF()
    layer = torch.compile(layer, fullgraph=True)
    data = torch.randn(10, 5, 4)
    out, s = layer(data)
    assert out.shape == (10, 5, 4)
    for x in s:
        assert x.shape == (5, 4)


def test_lif_feedforward_layer_backward():
    model = LIF()
    data = torch.ones(10, 12)
    out, _ = model(data)
    loss = out.sum()
    loss.backward()


def test_lif_feedforward_layer_backward_iteration():
    # Tests that gradient variables can be used in subsequent applications
    model = LIF()
    data = torch.ones(10, 6)
    out, s = model(data)
    out, _ = model(out, s)
    loss = out.sum()
    loss.backward()


def test_backward_model():
    model = SNNetwork()
    data = torch.ones(10, 12)
    out = model(data)
    loss = out.sum()
    loss.backward()


@pytest.mark.parametrize("sparse", [True, False])
def test_lif_datatype(sparse):
    input_tensor = torch.zeros(1, 2, dtype=torch.bool)
    p = LIFParameters().broadcast_to(input_tensor.shape)
    if sparse:
        input_tensor = input_tensor.to_sparse()
        p = p.to_sparse()

    for t in [LIFCell, LIF]:
        z, s = t(p)(input_tensor if t is LIFCell else input_tensor.unsqueeze(0))
        assert s.v.dtype == torch.float32
        assert s.v.is_sparse is sparse
        assert s.i.dtype == torch.float32
        assert s.i.is_sparse is sparse
        assert z.is_sparse is sparse


def test_lif_params_non_tensor():
    p = LIFParameters(
        tau_mem_inv=2, tau_syn_inv=1, v_leak=0.3, v_th=0.5, v_reset=0, alpha=20
    )
    LIFCell(p=p)(torch.ones(2, 3))
    LIF(p=p)(torch.ones(2, 3))
