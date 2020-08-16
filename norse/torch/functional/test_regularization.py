import torch 

from . import lif
from . import regularization

def regularisation_default_test():
    x = torch.ones(5, 10)
    s = lif.LIFState(torch.ones(10), torch.ones(10), torch.ones(10))
    z, s = lif.lif_feed_forward_step(x, s)
    zr, sr, rs = regularization.regularize_step(z, s)
    assert torch.all(torch.eq(z, zr))
    assert torch.all(torch.eq(s.v, sr.v))
    assert rs['num_spikes'] == 0
    z, s = lif.lif_feed_forward_step(x, s)
    zr, sr, rs = regularization.regularize_step(z, s)
    assert rs['num_spikes'] == 50