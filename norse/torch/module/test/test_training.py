"""
Tests that the training of Norse modules stays intact, for instance
that gradients are properly propagated
"""

import torch

from norse.torch.module.encode import PoissonEncoder
from norse.torch.module.lif import LIFRecurrentCell


class SNNetwork(torch.nn.Module):
    def __init__(self):
        super(SNNetwork, self).__init__()
        self.encoder = PoissonEncoder(10, f_max=1000)
        self.l0 = LIFRecurrentCell(12, 6)
        self.l1 = LIFRecurrentCell(6, 1)
        self.s0 = self.s1 = None

    def forward(self, input_tensor):
        spike_ts = self.encoder(input_tensor)
        spikes = None
        for spikes in spike_ts:
            spikes, self.s0 = self.l0(spikes, self.s0)
            spikes, self.s1 = self.l1(spikes, self.s1)
        return spikes


def test_optimize_model():
    model = SNNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=1)
    optimizer.zero_grad()
    input_weights = model.l0.input_weights.clone()
    recurrent_weights = model.l1.recurrent_weights.clone()
    data = torch.ones(1, 12)
    out = model(data)
    loss = out.sum()
    loss.backward()
    optimizer.step()
    assert not torch.all(torch.eq(input_weights, model.l0.input_weights))
    assert not torch.all(torch.eq(recurrent_weights, model.l0.recurrent_weights))
