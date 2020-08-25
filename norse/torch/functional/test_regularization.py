import torch

from . import lif
from . import regularization


def regularisation_spikes_test():
    x = torch.ones(5, 10)
    s = lif.LIFState(torch.ones(10), torch.ones(10), torch.ones(10))
    z, s = lif.lif_feed_forward_step(x, s)
    zr, rs = regularization.regularize_step(z, s)
    assert torch.all(torch.eq(z, zr))
    assert rs == 0
    z, s = lif.lif_feed_forward_step(x, s)
    zr, rs = regularization.regularize_step(z, s)
    assert rs == 50


def regularisation_voltage_test():
    x = torch.ones(5, 10)
    s = lif.LIFState(torch.ones(10), torch.ones(10), torch.ones(10))
    z, s = lif.lif_feed_forward_step(x, s)
    zr, rs = regularization.regularize_step(
        z, s, accumulator=regularization.voltage_accumulator
    )
    assert torch.all(torch.eq(z, zr))
    assert torch.all(torch.eq(s.v, rs))
