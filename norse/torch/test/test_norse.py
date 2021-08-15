"""Tests general uses of the library"""

import norse.torch as snn
from norse.torch.functional.lif import LIFParameters


def test_general_import():
    snn.LIFCell()


def test_absolute_import():
    LIFParameters()
