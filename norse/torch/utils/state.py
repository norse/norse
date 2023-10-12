import inspect

import torch


def _is_module_stateful(module: torch.nn.Module):
    """Tests whether a module uses state"""
    signature = inspect.signature(module.forward)
    return "state" in signature.parameters or isinstance(module, torch.nn.RNNBase)
