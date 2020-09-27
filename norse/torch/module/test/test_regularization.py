import torch

from norse.torch.module.lif import LIFFeedForwardCell
from norse.torch.module.regularization import RegularizationCell


def test_regularization_module():
    cell = LIFFeedForwardCell()  # 2 -> 4
    r = RegularizationCell()  # Defaults to spike counting
    data = torch.ones(5, 2) + 10  # Batch size of 5
    z, s = cell(data)
    z, rs = r(z, s)
    assert rs == 0
    z, s = cell(data, s)
    z, rs = r(z, s)
    assert rs == 10
    assert r.state == 10
