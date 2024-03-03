import onnx
import torch
import norse.torch as snn


def test_export_onnx_li():
    p = snn.LIParameters(tau_syn_inv=torch.as_tensor(42))
    net = snn.SequentialState(snn.LICell(p))
    inp = torch.randn(2, 1, 10)
    torch.onnx.export(net, inp, "snn_li.onnx")

    loaded = onnx.load("snn_li.onnx")
    onnx.checker.check_model(loaded)


def test_export_onnx_lif():
    p = snn.LIFParameters(tau_syn_inv=torch.as_tensor(42), v_th=torch.as_tensor(0.6))
    net = snn.SequentialState(snn.LIFCell(p))
    inp = torch.randn(2, 1, 10)
    torch.onnx.export(net, inp, "snn_lif.onnx")
