import torch
import pytest

from norse.torch.module.izhikevich import (
    IzhikevichCell,
    IzhikevichRecurrentCell,
    Izhikevich,
    IzhikevichRecurrent,
)
from norse.torch.functional import izhikevich
from norse.torch.functional.izhikevich import IzhikevichSpikingBehavior

list_method = [
    izhikevich.tonic_spiking,
    izhikevich.phasic_spiking,
    izhikevich.tonic_bursting,
    izhikevich.phasic_bursting,
    izhikevich.mixed_mode,
    izhikevich.spike_frequency_adaptation,
    izhikevich.class_1_exc,
    izhikevich.class_2_exc,
    izhikevich.spike_latency,
    izhikevich.subthreshold_oscillation,
    izhikevich.resonator,
    izhikevich.integrator,
    izhikevich.rebound_spike,
    izhikevich.rebound_burst,
    izhikevich.threshold_variability,
    izhikevich.bistability,
    izhikevich.dap,
    izhikevich.accomodation,
    izhikevich.inhibition_induced_spiking,
    izhikevich.inhibition_induced_bursting,
]


class SNNetwork(torch.nn.Module):
    def __init__(self, spiking_method: IzhikevichSpikingBehavior):
        super(SNNetwork, self).__init__()
        self.spiking_method = spiking_method
        self.l0 = Izhikevich(spiking_method)
        self.l1 = Izhikevich(spiking_method)
        self.s0 = self.s1 = None

    def forward(self, spikes):
        spikes, self.s0 = self.l0(spikes, self.s0)
        _, self.s1 = self.l1(spikes, self.s1)
        return self.s1.v.squeeze()


@pytest.mark.parametrize("spiking_method", list_method)
def test_izhikevich_cell(spiking_method):
    shape = (5, 2)
    data = torch.randn(shape)
    cell = IzhikevichCell(spiking_method)
    out, s = cell(data)

    for x in s:
        assert x.shape == (5, 2)
    assert out.shape == (5, 2)


@pytest.mark.parametrize("spiking_method", list_method)
def test_izhikevich_recurrent_cell(spiking_method):
    cell = IzhikevichRecurrentCell(2, 4, spiking_method)
    data = torch.randn(5, 2)
    out, s = cell(data)

    for x in s:
        assert x.shape == (5, 4)
    assert out.shape == (5, 4)


@pytest.mark.parametrize("spiking_method", list_method)
def test_izhikevich_recurrent_cell_autapses(spiking_method):
    cell = IzhikevichRecurrentCell(
        2,
        2,
        spiking_method,
        autapses=True,
        recurrent_weights=torch.ones(2, 2) * 0.01,
        dt=0.0001,
    )
    assert not torch.allclose(
        torch.zeros(2),
        (cell.recurrent_weights * torch.eye(*cell.recurrent_weights.shape)).sum(0),
    )
    s1 = izhikevich.IzhikevichRecurrentState(
        z=torch.ones(1, 2), v=torch.zeros(1, 2), u=torch.zeros(1, 2)
    )
    _, s_full = cell(torch.zeros(1, 2), s1)
    s2 = izhikevich.IzhikevichRecurrentState(
        z=torch.tensor([[0, 1]], dtype=torch.float32),
        v=torch.zeros(1, 2),
        u=torch.zeros(1, 2),
    )
    _, s_part = cell(torch.zeros(1, 2), s2)
    assert not s_full.v[0, 0] == s_part.v[0, 0]


@pytest.mark.parametrize("spiking_method", list_method)
def test_izhikevich_recurrent_cell_no_autapses(spiking_method):
    cell = IzhikevichRecurrentCell(2, 2, spiking_method, autapses=False)
    assert (
        cell.recurrent_weights * torch.eye(*cell.recurrent_weights.shape)
    ).sum() == 0

    s1 = izhikevich.IzhikevichRecurrentState(
        z=torch.ones(1, 2), v=torch.zeros(1, 2), u=torch.zeros(1, 2)
    )
    _, s_full = cell(torch.zeros(1, 2), s1)
    s2 = izhikevich.IzhikevichRecurrentState(
        z=torch.tensor([[0, 1]], dtype=torch.float32),
        v=torch.zeros(1, 2),
        u=torch.zeros(1, 2),
    )
    _, s_part = cell(torch.zeros(1, 2), s2)

    assert s_full.v[0, 0] == s_part.v[0, 0]


@pytest.mark.parametrize("spiking_method", list_method)
def test_izhikevich_in_time(spiking_method):
    layer = Izhikevich(spiking_method)
    data = torch.randn(10, 5, 2)
    out, _ = layer(data)

    assert out.shape == (10, 5, 2)


@pytest.mark.parametrize("spiking_method", list_method)
def test_izhikevich_recurrent_sequence(spiking_method):
    l1 = IzhikevichRecurrent(8, 6, spiking_method)
    l2 = IzhikevichRecurrent(6, 4, spiking_method)
    l3 = IzhikevichRecurrent(4, 1, spiking_method)
    z = torch.ones(10, 1, 8)
    z, s1 = l1(z)
    z, s2 = l2(z)
    z, s3 = l3(z)
    assert s1.v.shape == (1, 6)
    assert s2.v.shape == (1, 4)
    assert s3.v.shape == (1, 1)
    assert z.shape == (10, 1, 1)


@pytest.mark.parametrize("spiking_method", list_method)
def test_izhikevich_feedforward_cell_backward(spiking_method):
    # Tests that gradient variables can be used in subsequent applications
    cell = IzhikevichCell(spiking_method)
    data = torch.randn(5, 4)
    out, s = cell(data)
    out, _ = cell(out, s)
    loss = out.sum()
    loss.backward()


@pytest.mark.parametrize("spiking_method", list_method)
def test_izhikevich_recurrent_cell_backward(spiking_method):
    # Tests that gradient variables can be used in subsequent applications
    cell = IzhikevichRecurrentCell(4, 4, spiking_method)
    data = torch.randn(5, 4)
    out, s = cell(data)
    out, _ = cell(out, s)
    loss = out.sum()
    loss.backward()


@pytest.mark.parametrize("spiking_method", list_method)
def test_izhikevich_feedforward_layer(spiking_method):
    layer = Izhikevich(spiking_method)
    data = torch.randn(10, 5, 4)
    out, s = layer(data)
    assert out.shape == (10, 5, 4)
    for x in s:
        assert x.shape == (5, 4)


@pytest.mark.parametrize("spiking_method", list_method)
def test_izhikevich_feedforward_layer_backward(spiking_method):
    model = Izhikevich(spiking_method)
    data = torch.ones(10, 12)
    out, _ = model(data)
    loss = out.sum()
    loss.backward()


@pytest.mark.parametrize("spiking_method", list_method)
def test_izhikevich_recurrent_layer_backward_iteration(spiking_method):
    # Tests that gradient variables can be used in subsequent applications
    model = IzhikevichRecurrent(6, 6, spiking_method)
    data = torch.ones(10, 6)
    out, s = model(data)
    out, _ = model(out, s)
    loss = out.sum()
    loss.backward()


@pytest.mark.parametrize("spiking_method", list_method)
def test_izhikevich_recurrent_layer_backward(spiking_method):
    model = IzhikevichRecurrent(6, 6, spiking_method)
    data = torch.ones(10, 6)
    out, _ = model(data)
    loss = out.sum()
    loss.backward()


@pytest.mark.parametrize("spiking_method", list_method)
def test_izhikevich_feedforward_layer_backward_iteration(spiking_method):
    # Tests that gradient variables can be used in subsequent applications
    model = Izhikevich(spiking_method)
    data = torch.ones(10, 6)
    out, s = model(data)
    out, _ = model(out, s)
    loss = out.sum()
    loss.backward()


@pytest.mark.parametrize("spiking_method", list_method)
def test_backward_model(spiking_method):
    model = SNNetwork(spiking_method)
    data = torch.ones(10, 12)
    out = model(data)
    loss = out.sum()
    loss.backward()
