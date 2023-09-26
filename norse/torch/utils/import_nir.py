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
        """The parameter {} is expected to be set to {}, but was found to diverge (with mean {}).
Please read the Norse documentation for more information.
This warning can be turned off by setting from_nir(..., ignore_warnings=True)
"""
    )


def _is_identity(array: np.ndarray, expected_value: float):
    return np.allclose(array, expected_value)


def _import_norse_module(
    node: nir.NIRNode, ignore_warnings: bool = False
) -> torch.nn.Module:
    if isinstance(node, nir.Affine):
        return torch.nn.Linear(node.weight.shape[1], node.weight.shape[0])
    if isinstance(node, nir.Conv2d):
        return torch.nn.Conv2d(
            in_channels=node.weight.shape[1],
            out_channels=node.weight.shape[0],
            kernel_size=node.weight.shape[-2:],
            stride=node.stride,
            padding=node.padding,
            dilation=node.dilation,
            groups=node.groups,
        )
    if isinstance(node, nir.Linear):
        return torch.nn.Linear(
            node.weight.shape[1], torch.zeros_like(node.weight.shape[0])
        )
    if isinstance(node, nir.IF):
        if not _is_identity(node.r, 1) and not ignore_warnings:
            _log_warning("r", 1, node.r)
        return iaf.IAFCell(iaf.IAFParameters(v_th=node.v_threshold))
    if isinstance(node, nir.CubaLIF):
        if not _is_identity(node.r, 1) and not ignore_warnings:
            _log_warning("r", 1, node.r)
        return lif.LIFCell(
            lif.LIFParameters(
                tau_mem_inv=1 / node.tau_mem,  # Invert time constant
                tau_syn_inv=1 / node.tau_syn,  # Invert time constant
                v_th=node.v_threshold,
                v_leak=node.v_leak,
            )
        )
    if isinstance(node, nir.LIF):
        if not _is_identity(node.r, 1) and not ignore_warnings:
            _log_warning("r", 1, node.r)
        return lif_box.LIFBoxCell(
            lif_box.LIFBoxParameters(
                tau_mem_inv=1 / node.tau,  # Invert time constant
                v_th=node.v_threshold,
                v_leak=node.v_leak,
            )
        )


def from_nir(node: nir.NIRNode, ignore_warnings: bool = False) -> torch.nn.Module:
    return nirtorch.load(node, _import_norse_module)
