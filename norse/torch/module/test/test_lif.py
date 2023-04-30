import pytest, platform
import torch

from norse.torch.functional.lif import LIFState, LIFParameters
from norse.torch.module.lif import (
    LIFCell,
    LIFRecurrent,
    LIF,
    LIFRecurrentCell,
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


@pytest.mark.skipif(
    not platform.system() == "Linux", reason="Only Linux supports torch.compile"
)
def test_lif_cell_feedforward_compile():
    layer = LIFCell()
    layer = torch.compile(layer)
    data = torch.randn(5, 4)
    out, s = layer(data)
    assert out.shape == (5, 4)
    for x in s:
        assert x.shape == (5, 4)


def test_lif_recurrent_cell():
    cell = LIFRecurrentCell(2, 4)
    data = torch.randn(5, 2)
    out, s = cell(data)

    for x in s:
        assert x.shape == (5, 4)
    assert out.shape == (5, 4)


@pytest.mark.skipif(
    not platform.system() == "Linux", reason="Only Linux supports torch.compile"
)
def test_lif_recurrent_cell_compile():
    cell = LIFRecurrentCell(2, 4)
    cell = torch.compile(cell)
    data = torch.randn(5, 2)
    out, s = cell(data)

    for x in s:
        assert x.shape == (5, 4)
    assert out.shape == (5, 4)


def test_lif_recurrent_cell_autapses():
    cell = LIFRecurrentCell(2, 2, autapses=True)
    assert not torch.allclose(
        torch.zeros(2),
        (cell.recurrent_weights * torch.eye(*cell.recurrent_weights.shape)).sum(0),
    )
    s1 = LIFState(z=torch.ones(1, 2), v=torch.zeros(1, 2), i=torch.zeros(1, 2))
    z, s_full = cell(torch.zeros(1, 2), s1)
    s2 = LIFState(
        z=torch.tensor([[0, 1]], dtype=torch.float32),
        v=torch.zeros(1, 2),
        i=torch.zeros(1, 2),
    )
    z, s_part = cell(torch.zeros(1, 2), s2)

    assert not s_full.i[0, 0] == s_part.i[0, 0]


def test_lif_recurrent_cell_no_autapses():
    cell = LIFRecurrentCell(2, 2, autapses=False)
    assert (
        cell.recurrent_weights * torch.eye(*cell.recurrent_weights.shape)
    ).sum() == 0

    s1 = LIFState(z=torch.ones(1, 2), v=torch.zeros(1, 2), i=torch.zeros(1, 2))
    z, s_full = cell(torch.zeros(1, 2), s1)
    s2 = LIFState(
        z=torch.tensor([[0, 1]], dtype=torch.float32),
        v=torch.zeros(1, 2),
        i=torch.zeros(1, 2),
    )
    z, s_part = cell(torch.zeros(1, 2), s2)

    assert s_full.i[0, 0] == s_part.i[0, 0]


def test_lif_in_time():
    layer = LIF()
    data = torch.randn(10, 5, 2)
    out, _ = layer(data)

    assert out.shape == (10, 5, 2)


def test_lif_recurrent_sequence():
    l1 = LIFRecurrent(8, 6)
    l2 = LIFRecurrent(6, 4)
    l3 = LIFRecurrent(4, 1)
    z = torch.ones(10, 1, 8)
    z, s1 = l1(z)
    z, s2 = l2(z)
    z, s3 = l3(z)
    assert s1.v.shape == (1, 6)
    assert s2.v.shape == (1, 4)
    assert s3.v.shape == (1, 1)
    assert z.shape == (10, 1, 1)


def test_lif_feedforward_cell_backward():
    # Tests that gradient variables can be used in subsequent applications
    cell = LIFCell()
    data = torch.randn(5, 4)
    out, s = cell(data)
    out, _ = cell(out, s)
    loss = out.sum()
    loss.backward()


def test_lif_recurrent_cell_backward():
    # Tests that gradient variables can be used in subsequent applications
    cell = LIFRecurrentCell(4, 4)
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
    not platform.system() == "Linux", reason="Only Linux supports torch.compile"
)
def test_lif_feedforward_layer_compile():
    layer = LIF()
    layer = torch.compile(layer)
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


def test_lif_recurrent_layer_backward_iteration():
    # Tests that gradient variables can be used in subsequent applications
    model = LIFRecurrent(6, 6)
    data = torch.ones(10, 6)
    out, s = model(data)
    out, _ = model(out, s)
    loss = out.sum()
    loss.backward()


def test_lif_recurrent_layer_backward():
    model = LIFRecurrent(6, 6)
    data = torch.ones(10, 6)
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


def test_lif_datatype():
    for sparse in [True, False]:
        input_tensor = torch.zeros(1, 2, dtype=torch.bool)
        if sparse:
            input_tensor = input_tensor.to_sparse()

        p = LIFParameters()
        for t in [LIFCell, LIF]:
            z, s = t(p)(input_tensor)
            assert s.v.dtype == torch.float32
            assert s.i.dtype == torch.float32
            assert z.is_sparse is sparse

        # Recurrent layers only supports floats due to the linear layer
        input_tensor = input_tensor.float()
        z, s = LIFRecurrentCell(2, 1, p)(input_tensor)
        assert s.v.dtype == torch.float32
        assert s.i.dtype == torch.float32
        assert z.is_sparse is sparse

        input_tensor = input_tensor.unsqueeze(1)
        z, s = LIFRecurrent(2, 1, p)(input_tensor)
        assert s.v.dtype == torch.float32
        assert s.i.dtype == torch.float32
        assert z.is_sparse is sparse


def test_lif_params_non_tensor():
    p = LIFParameters(
        tau_mem_inv=2, tau_syn_inv=1, v_leak=0.3, v_th=0.5, v_reset=0, alpha=20
    )
    LIFCell(p=p)(torch.ones(2, 3))
    LIF(p=p)(torch.ones(2, 3))
