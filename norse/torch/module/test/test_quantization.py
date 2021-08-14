"""
Tests that Norse correctly supports quantization
"""
from norse.torch.module.lif import LIFCell, LIFRecurrentCell
import torch


class SNNModel(torch.nn.Module):
    def __init__(self, snn):
        super().__init__()
        self.lin = torch.nn.Linear(2, 1)
        self.snn = snn

    def forward(self, x, s):
        return self.snn(self.lin(x), s)


def test_quantize_dynamic_cell():
    m_fp32 = SNNModel(LIFCell())
    m_int8 = torch.quantization.quantize_dynamic(
        m_fp32, {torch.nn.Linear, LIFCell}, dtype=torch.qint8
    )
    assert isinstance(m_int8.lin, torch.nn.quantized.dynamic.modules.linear.Linear)
    data = torch.randn(2, 2)
    m_int8(data)


def test_quantized_dynamic_rnn():
    m_fp32 = SNNModel(LIFRecurrentCell)
    m_int8 = torch.quantization.quantize_dynamic(
        m_fp32, {torch.nn.Linear, LIFRecurrentCell}, dtype=torch.qint8
    )
    
