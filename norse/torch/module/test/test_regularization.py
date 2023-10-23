import torch

from norse.torch.module.lif import LIFCell
from norse.torch.module.regularization import RegularizationCell


def test_regularization_module():
    cell = LIFCell()
    r = RegularizationCell()  # Defaults to spike counting
    data = torch.ones(5, 2) + 10  # Batch size of 5
    z, s = cell(data)
    z, rs = r(z, s)
    assert z.shape == (5, 2)
    assert rs == 10
    z, s = cell(data, s)
    z, rs = r(z, s)
    assert rs == 20
    assert r.state == 20
