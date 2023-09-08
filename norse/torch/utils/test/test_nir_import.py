import nir
import nirtorch
import torch

import norse.torch as norse


def test_import_linear():
    n = nir.NIRGraph.from_list(
        nir.Input(torch.tensor([2])),
        nir.Affine(torch.randn(2, 3), torch.randn(2)),
        nir.Output(torch.tensor([3])),
    )
    m = norse.from_nir(n)
    print(m)
    # assert isinstance(m, torch.nn.Linear)
