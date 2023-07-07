import torch
import norse.torch as snn
import norse.torch.utils.export as export


def test_sequential():
    m = snn.SequentialState(
        snn.LIFBoxCell(),
        torch.nn.Linear(10, 2),
        snn.LIBoxCell(),
        torch.nn.Linear(2, 1),
    )
    graph = export.to_nir(m, torch.randn(1, 10))
    assert len(graph.nodes) == 4
    assert graph.nodes[0].type == "LIF"
    assert graph.nodes[0].type == "Linear"
    assert graph.nodes[0].type == "LI"
    assert graph.nodes[3].type == "Linear"
    assert len(graph.edges) == 3
