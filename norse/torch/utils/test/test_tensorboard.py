import torch
import norse.torch as norse

from norse.torch.utils.tensorboard import tensorboard


class MockWriter:
    key = None
    spikes = None
    index = None

    def add_histogram(self, key, spikes, index):
        self.key = key
        self.spikes = spikes
        self.index = index

    def add_image(self, key, spikes, index):
        self.key = key
        self.spikes = spikes
        self.index = index

    def add_scalar(self, key, spikes, index):
        self.key = key
        self.spikes = spikes
        self.index = index


def test_activity_hook():
    cell = norse.LIFCell()
    writer = MockWriter()
    hook = tensorboard.hook_spike_activity_mean("lif", writer)
    cell.register_forward_hook(hook)
    s = None
    for _ in range(6):
        z, s = cell(torch.ones(2), s)

    assert z.max() > 0
    assert torch.eq(writer.spikes, z.mean())

    hook = tensorboard.hook_spike_activity_sum("lif", writer)
    cell.register_forward_hook(hook)
    s = None
    for _ in range(6):
        z, s = cell(torch.ones(2), s)

    assert z.max() > 0
    assert torch.eq(writer.spikes, z.sum())


def test_image_hook():
    cell = norse.LIFCell()
    writer = MockWriter()
    hook = tensorboard.hook_spike_image("lif", writer)
    cell.register_forward_hook(hook)
    s = None
    for _ in range(6):
        z, s = cell(torch.ones(2), s)

    assert z.max() > 0
    assert torch.all(torch.eq(writer.spikes, z))


def test_histogram_hook():
    cell = norse.LIFCell()
    writer = MockWriter()
    hook = tensorboard.hook_spike_histogram_mean("lif", writer)
    cell.register_forward_hook(hook)
    s = None
    for _ in range(6):
        z, s = cell(torch.ones(2), s)

    assert z.max() > 0
    assert torch.eq(writer.spikes, z.mean())

    hook = tensorboard.hook_spike_histogram_sum("lif", writer)
    cell.register_forward_hook(hook)
    s = None
    for _ in range(6):
        z, s = cell(torch.ones(2), s)

    assert z.max() > 0
    assert torch.eq(writer.spikes, z.sum())
