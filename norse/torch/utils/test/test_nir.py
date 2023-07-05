import torch
import norse.torch as snn
import norse.torch.utils.export as export


def test_sequential():
    m = snn.SequentialState(
        snn.LIFBoxCell(),
        torch.nn.Linear(10, 2),
        snn.LIFBoxCell(),
        torch.nn.Linear(2, 1),
    )
    graph = export.to_nir(m, torch.randn(1, 10))
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 3
