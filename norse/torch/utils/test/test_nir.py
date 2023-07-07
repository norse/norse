import torch
import nir

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
    assert isinstance(graph.nodes[0], nir.LIF)
    assert isinstance(graph.nodes[1], nir.Linear)
    assert isinstance(graph.nodes[2], nir.LI)
    assert isinstance(graph.nodes[3], nir.Linear)
    assert len(graph.edges) == 3

def test_linear():
    in_features = 2
    out_features = 3
    m = torch.nn.Linear(in_features, out_features, bias=False)
    m2 = torch.nn.Linear(in_features, out_features, bias=True)
    graph = export.to_nir(m, torch.randn(1, in_features))
    assert len(graph.nodes) == 1
    assert graph.nodes[0].weights.shape == (out_features, in_features)
    assert graph.nodes[0].bias.shape == m2.bias.shape


