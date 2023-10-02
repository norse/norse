from functools import partial
from numbers import Number
from typing import Union

import nir
import nirtorch
import numpy as np
import torch

import norse.torch.module.iaf as iaf
import norse.torch.module.lif_box as lif_box
import norse.torch.module.lif as lif

import logging


def _log_warning(parameter_name: str, expected_value: str, actual_value: str):
    logging.warning(
        f"""The parameter {parameter_name} is expected to be set to {expected_value}, but was found to diverge (with mean {actual_value}).
Please read the Norse documentation for more information.
This warning can be turned off by setting from_nir(..., ignore_warnings=True)
"""
    )


def _is_identity(array: np.ndarray, expected_value: float):
    return np.allclose(array, expected_value)


def _to_tensor(tensor: Union[np.ndarray, torch.Tensor]):
    if isinstance(tensor, torch.Tensor):
        return tensor
    if isinstance(tensor, Number):
        return torch.as_tensor(tensor)
    return torch.from_numpy(tensor)


def _import_norse_module(
    node: nir.NIRNode, ignore_warnings: bool = False
) -> torch.nn.Module:
    if isinstance(node, nir.Affine):
        module = torch.nn.Linear(node.weight.shape[1], node.weight.shape[0])
        module.weight.data = _to_tensor(node.weight)
        module.bias.data = _to_tensor(node.bias)
        return module
    if isinstance(node, nir.Conv2d):
        module = torch.nn.Conv2d(
            in_channels=node.weight.shape[1],
            out_channels=node.weight.shape[0],
            kernel_size=node.weight.shape[-2:],
            stride=node.stride,
            padding=node.padding,
            dilation=node.dilation,
            groups=node.groups,
        )
        module.weight.data = _to_tensor(node.weight)
        module.bias.data = _to_tensor(node.bias)
        return module
    if isinstance(node, nir.Flatten):
        return torch.nn.Flatten(node.start_dim, node.end_dim)
    if isinstance(node, nir.Linear):
        module = torch.nn.Linear(
            node.weight.shape[1],
            torch.zeros_like(node.weight.shape[0]),
            bias=False,
        )
        module.weight.data = _to_tensor(node.weight)
        return module
    if isinstance(node, nir.IF):
        if not _is_identity(node.r, 1) and not ignore_warnings:
            _log_warning("r", 1, node.r.mean())
        return iaf.IAFCell(iaf.IAFParameters(v_th=_to_tensor(node.v_threshold)))
    if isinstance(node, nir.CubaLIF):
        if not _is_identity(node.r, 1) and not ignore_warnings:
            _log_warning("r", 1, node.r.mean())
        return lif.LIFCell(
            lif.LIFParameters(
                tau_mem_inv=1 / _to_tensor(node.tau_mem),  # Invert time constant
                tau_syn_inv=1 / _to_tensor(node.tau_syn),  # Invert time constant
                v_th=_to_tensor(node.v_threshold),
                v_leak=_to_tensor(node.v_leak),
            )
        )
    if isinstance(node, nir.LIF):
        if not _is_identity(node.r, 1) and not ignore_warnings:
            _log_warning("r", 1, _to_tensor(node.r).mean())
        return lif_box.LIFBoxCell(
            lif_box.LIFBoxParameters(
                tau_mem_inv=1 / _to_tensor(node.tau),  # Invert time constant
                v_th=_to_tensor(node.v_threshold),
                v_leak=_to_tensor(node.v_leak),
            )
        )
    if isinstance(node, nir.SumPool2d):
        if not np.allclose(node.padding, 0.0) and not ignore_warnings:
            _log_warning("padding", 0, node.padding.mean())
        return torch.nn.LPPool2d(
            norm_type=1,
            kernel_size=tuple(node.kernel_size),
            stride=tuple(node.stride),
        )


def from_nir(node: nir.NIRNode, ignore_warnings: bool = False) -> torch.nn.Module:
    """Converts a NIR graph to a Norse module.

    Example:
    >>> import nir
    >>> import norse.torch
    >>> g = nir.read("my_graph.nir")
    >>> module = norse.torch.from_nir(g)

    Arguments:
        node (nir.NIRNode): The root node of the NIR graph to convert.
        ignore_warnings (bool): Whether to ignore warnings about diverging or unsupported parameters.
    """
    return nirtorch.load(
        node, partial(_import_norse_module, ignore_warnings=ignore_warnings)
    )
