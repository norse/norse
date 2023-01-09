import torch
import norse.torch as norse


def test_sequential_synops():
    model = norse.SequentialState(torch.nn.Linear(1, 10, bias=False), norse.LIF())
    norse.hooks.register_synops(model)
    assert hasattr(model[0], norse.hooks._SYNOPS_BUFFER)
    assert hasattr(model[1], norse.hooks._SYNOPS_BUFFER)

    lin = model[0](torch.ones(1, 1))
    _, s = model(torch.ones(1, 1))

    assert norse.hooks.count_synops_recursively(model) == 10 + lin.sum()
