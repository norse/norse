"""
Utility module for Norse.
"""
try:
    import torch
    import norse_op

    IS_OPS_LOADED = True
    del torch  # Unload torch again to allow importing norse.torch
except ModuleNotFoundError:
    IS_OPS_LOADED = False
