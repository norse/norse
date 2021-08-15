"""
Tests that Norse correctly supports quantization
"""
import torch
from norse.torch.module.lif import LIFCell, LIFRecurrentCell, LIFFeedForwardState


class SNNModelDynamic(torch.nn.Module):
    def __init__(self, snn):
        super().__init__()
        self.lin = torch.nn.Linear(2, 1)
        self.snn = snn

    def forward(self, x, s=None):
        return self.snn(self.lin(x), s)


class SNNModelStatic(torch.nn.Module):
    def __init__(self, snn):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.snn = snn
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x, s=None):
        return self.dequant(self.snn(self.conv(self.quant(x)), s))


def test_quantize_dynamic_cell():
    m_fp32 = SNNModelDynamic(LIFCell())
    m_int8 = torch.quantization.quantize_dynamic(
        m_fp32, {torch.nn.Linear, LIFCell}, dtype=torch.qint8
    )
    assert isinstance(m_int8.lin, torch.nn.quantized.dynamic.modules.linear.Linear)
    data = torch.randn(2, 2)
    m_int8(data)


def test_quantized_dynamic_rnn():
    m_fp32 = SNNModelDynamic(LIFRecurrentCell(1, 1))
    m_int8 = torch.quantization.quantize_dynamic(
        m_fp32, {torch.nn.Linear, LIFRecurrentCell}, dtype=torch.qint8
    )
    assert isinstance(m_int8.lin, torch.nn.quantized.dynamic.modules.linear.Linear)
    assert isinstance(m_int8.snn.linear_in, torch.nn.quantized.dynamic.modules.linear.Linear)
    assert isinstance(m_int8.snn.linear_rec, torch.nn.quantized.dynamic.modules.linear.Linear)
    data = torch.randn(2, 2)
    m_int8(data)