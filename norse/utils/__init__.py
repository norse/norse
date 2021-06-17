"""
Utility module for Norse.
"""
import sys

try:
    import torch
    import norse_op

    IS_OPS_LOADED = True
except ModuleNotFoundError:
    IS_OPS_LOADED = False
