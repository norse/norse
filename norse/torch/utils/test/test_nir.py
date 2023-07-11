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
    assert len(graph.nodes) == 6 # 4 + 2 for input and output
    assert isinstance(graph.nodes[0], nir.Input)
    assert isinstance(graph.nodes[1], nir.LIF)
    assert isinstance(graph.nodes[2], nir.Affine)
    assert isinstance(graph.nodes[3], nir.LI)
    assert isinstance(graph.nodes[4], nir.Affine)
    assert isinstance(graph.nodes[5], nir.Output)
    assert len(graph.edges) == 5

def test_linear():
    in_features = 2
    out_features = 3
    m = torch.nn.Linear(in_features, out_features, bias=False)
    m2 = torch.nn.Linear(in_features, out_features, bias=True)
    graph = export.to_nir(m, torch.randn(1, in_features))
    assert len(graph.nodes) == 3
    assert graph.nodes[1].weight.shape == (out_features, in_features)
    assert graph.nodes[1].bias.shape == m2.bias.shape


