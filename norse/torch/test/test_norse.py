"""Tests general uses of the library"""

import norse.torch as norse
from norse.torch.functional.lif import LIFParameters


def test_general_import():
    norse.LIFCell()


def test_absolute_import():
    LIFParameters()
